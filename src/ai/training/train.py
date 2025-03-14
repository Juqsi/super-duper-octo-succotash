import gc
import time
import torch
import os
import json
import numpy as np
import torchvision.models as models
import glob
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm
import config
from utils import save_checkpoint, load_checkpoint
from torchvision.models import ResNet50_Weights


# Focal Loss für besseres Lernen seltener Klassen
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


# Mixup für verbesserte Generalisierung
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Datentransformation mit Augmentation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Lade Datensätze
selected_classes = ['1356126', '1363128', '1356022', '1357330', '1355978', '1363740', '1364172', '1355937', '1361656',
                    '1363021', '1385937', '1356421', '1358094', '1384485', '1393614']


def load_dataset(root_dir):
    dataset = datasets.ImageFolder(root_dir, transform=transform)
    dataset.samples = [s for s in dataset.samples if dataset.classes[s[1]] in selected_classes]
    class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
    dataset.class_to_idx = class_to_idx
    dataset.samples = [(s[0], class_to_idx[dataset.classes[s[1]]]) for s in dataset.samples]
    dataset.classes = selected_classes
    return dataset


train_dataset = load_dataset(config.TRAIN_DIR)
val_dataset = load_dataset(config.VAL_DIR)
test_dataset = load_dataset(config.TEST_DIR)

# Weighted Sampling für Klassenbalance
class_counts = [sum(1 for _, label in train_dataset.samples if label == idx) for idx in range(len(selected_classes))]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[label] for _, label in train_dataset.samples if label < len(selected_classes)]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=config.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)


# Modell (Transfer Learning mit ResNet50)
if os.path.exists(config.PRETRAINED_MODEL_PATH):
    print("Lade vortrainiertes Modell aus lokalem Speicher...")
    model = models.resnet50()
    model.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH))
else:
    print("Lade vortrainiertes Modell von torchvision (dauert länger)...")
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    torch.save(model.state_dict(), config.PRETRAINED_MODEL_PATH)  # Modell lokal speichern

model.fc = nn.Linear(model.fc.in_features, len(selected_classes))
model = model.to(config.DEVICE)

# Optimierung & Loss
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
criterion = FocalLoss(alpha=0.25, gamma=2.0)
scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

# Falls Fortsetzung des Trainings
start_epoch = 0
if config.RESUME_TRAINING:
    checkpoint_path = config.CHECKPOINT_PATH.format(config.LAST_EPOCH)
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, config.DEVICE)

# Training
train_accuracy_list = []
accuracy_list = []
epoch_times = []
training_start_time = time.time()

for epoch in range(start_epoch, config.EPOCHS):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    if epoch >= config.MIXUP_DISABLE_EPOCH and config.USE_MIXUP:
        config.USE_MIXUPIU = False  # Mixup deaktivieren ab Epoche
        print(f"🔄 Mixup deaktiviert ab Epoche {epoch + 1}")

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()

        if config.USE_MIXUP:
            # Mixup aktiv
            inputs, y_a, y_b, lam = mixup_data(inputs, labels)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            # Standard-Training ohne Mixup
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_accuracy_list.append(epoch_acc)

    # Validierung
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total
    accuracy_list.append(val_acc)

    print(
        f"Epoch [{epoch + 1}/{config.EPOCHS}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    scheduler.step()
    save_checkpoint(model, optimizer, epoch, config.CHECKPOINT_PATH.format(epoch))
    epoch_times.append(time.time() - epoch_start_time)

total_training_time = time.time() - training_start_time
