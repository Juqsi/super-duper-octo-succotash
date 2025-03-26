import os
import gc
import time
import json
import torch
import torchvision.models as models
from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights
from torchvision.transforms import RandAugment
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import ft_config as config
from src.ai.training.utils import save_checkpoint
from src.ai.training.stats import save_confusion_matrix, plot_training_progress

# Merge Map laden
with open(config.MERGE_MAP_PATH, "r") as f:
    merge_map = json.load(f)

# Alle Originalklassen im Datensatz
all_classes = sorted([c for c in os.listdir(config.TRAIN_DIR) if not c.startswith('.')])

# Alle Klassen, die NICHT gemerged wurden, zu sich selbst mappen
full_merge_map = {cls: merge_map.get(cls, cls) for cls in all_classes}

# Alle Zielklassen (nach Merge)
selected_classes = sorted(set(full_merge_map.values()))

# Index-Zuweisung
class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
old_to_new_index = {old: class_to_idx[new] for old, new in full_merge_map.items()}

# --- Datentransformation ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Dataset laden und Labels mergen ---
def load_dataset(root_dir):
    dataset = datasets.ImageFolder(root_dir, transform=transform)
    new_samples = []
    for path, label in dataset.samples:
        old_class = dataset.classes[label]
        if old_class in old_to_new_index:
            new_label = old_to_new_index[old_class]
            new_samples.append((path, new_label))

    dataset.samples = new_samples
    dataset.targets = [label for _, label in new_samples]
    dataset.classes = selected_classes
    dataset.class_to_idx = class_to_idx
    return dataset

train_dataset = load_dataset(config.TRAIN_DIR)
val_dataset = load_dataset(config.VAL_DIR)

# --- Weighted Sampling ---
class_counts = [sum(1 for _, label in train_dataset.samples if label == i) for i in range(len(selected_classes))]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=config.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

print(len(selected_classes))
print(selected_classes)

# --- Modell vorbereiten ---
model = models.resnet50()

# 1️⃣ Checkpoint laden
start_epoch = 0
if config.RESUME_TRAINING:
    checkpoint_path = config.CHECKPOINT_PATH.format(config.LAST_EPOCH)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    fc_shape = checkpoint['model_state_dict']['fc.weight'].shape[0]
    model.fc = nn.Linear(model.fc.in_features, fc_shape)  # FC auf alte Größe setzen
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"✅ Resume von {checkpoint_path}")
    start_epoch = config.LAST_EPOCH + 1
else:
    checkpoint = torch.load(config.PREVIOUS_CHECKPOINT, map_location=config.DEVICE)
    fc_shape = checkpoint['model_state_dict']['fc.weight'].shape[0]
    model.fc = nn.Linear(model.fc.in_features, fc_shape)  # FC auf alte Größe setzen
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"📦 Vorheriger Checkpoint geladen: {config.PREVIOUS_CHECKPOINT}")

# 2️⃣ Jetzt neue FC-Schicht setzen (für Merged Classes)
model.fc = nn.Linear(model.fc.in_features, len(selected_classes))
model = model.to(config.DEVICE)

# 3️⃣ Trainierbare Layer festlegen
for param in model.parameters():
    param.requires_grad = False
for block in [ model.layer3, model.layer4, model.fc]:
    for param in block.parameters():
        param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
criterion = nn.CrossEntropyLoss()

# --- Training ---
train_acc_list, val_acc_list, top5_acc_list, epoch_times = [], [], [], []
print("🚀 Starte Finetuning...")
training_start_time = time.time()

for epoch in range(start_epoch, config.EPOCHS):
    model.train()
    epoch_start_time = time.time()
    gc.collect(); torch.cuda.empty_cache()
    correct, total, running_loss = 0, 0, 0.0

    tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", dynamic_ncols=True)

    for inputs, labels in tqdm_loader:
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 📊 Live-Update
        current_loss = running_loss / (total // config.BATCH_SIZE + 1)
        current_acc = correct / total
        tqdm_loader.set_postfix({
            "loss": f"{current_loss:.4f}",
            "acc": f"{current_acc:.4f}"
        })

    train_acc = correct / total
    train_acc_list.append(train_acc)

    # --- Validierung ---
    model.eval()
    val_correct, val_total, top5_correct, val_loss = 0, 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            top5_preds = outputs.topk(5, dim=1).indices
            top5_correct += sum(label in pred for pred, label in zip(top5_preds, labels))
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    top5_acc = top5_correct / val_total
    val_acc_list.append(val_acc)
    top5_acc_list.append(top5_acc)

    scheduler.step()
    train_loss = running_loss / len(train_loader)
    val_loss_avg = val_loss / len(val_loader)

    print(f"📊 Epoch {epoch + 1}: "
          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
          f"Val Loss={val_loss_avg:.4f}, Val Acc={val_acc:.4f}, Top-5 Acc={top5_acc:.4f}")

    save_checkpoint(model, optimizer, epoch, config.CHECKPOINT_PATH.format(epoch))
    if (epoch + 1) % 10 == 0 or epoch == 0:
        save_confusion_matrix(model, val_loader, selected_classes, config.DEVICE, epoch + 1, config.CHECKPOINT_DIR)
    epoch_times.append(time.time() - epoch_start_time)

# --- Speichern ---
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_path = os.path.join(config.MODEL_DIR, f"finetuned_model_{now}.pth")
torch.save(model.state_dict(), model_path)
config_path = os.path.join(config.MODEL_DIR, f"config_finetune_{now}.json")
with open(config_path, 'w') as f:
    json.dump({
        "CLASSES": selected_classes,
        "MERGE_MAP": merge_map,
        "VAL_ACC": val_acc_list,
        "TOP_5_ACC": top5_acc_list,
        "BEST_VAL_ACC": max(val_acc_list),
        "BEST_TOP5_ACC": max(top5_acc_list),
        "EPOCH_TIMES": epoch_times,
        "TOTAL_TIME": time.time() - training_start_time
    }, f, indent=4)

plot_training_progress(train_acc_list, val_acc_list, top5_acc_list, config.MODEL_DIR)
print(f"✅ Finetuning abgeschlossen. Modell gespeichert unter: {model_path}")
