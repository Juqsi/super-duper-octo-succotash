"""
Fine-tuning script for a ResNet50 model with class merging.

This script performs the following steps:

1. Load and merge classes based on a merge map.
2. Define data transformations and load the datasets.
3. Create DataLoaders with weighted sampling for balanced training.
4. Load a checkpoint (either to resume training or as a starting point).
5. Adapt the model by replacing the fully connected (FC) layer to match the merged classes.
6. Freeze parts of the model, allowing only specific layers (Layer 3, Layer 4, and FC) to be trained.
7. Perform the fine-tuning process, including training, validation, and metric logging.
8. Save the final model, configuration, and visualize the training progress.
"""
import json
import os
from datetime import datetime

import torch
import torchvision.models as models
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
from tqdm import tqdm

from . import ft_config as config

# --- Load Merge Map and Combine Classes ---
with open(config.MERGE_MAP_PATH, "r") as f:
    merge_map = json.load(f)

all_classes = sorted([c for c in os.listdir(config.TRAIN_DIR) if not c.startswith('.')])

full_merge_map = {cls: merge_map.get(cls, cls) for cls in all_classes}
selected_classes = sorted(set(full_merge_map.values()))

class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
old_to_new_index = {old: class_to_idx[new] for old, new in full_merge_map.items()}

# --- Data Transformation ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_dataset(root_dir):
    """
    Loads an ImageFolder dataset, merges labels according to the merge map, and adjusts class mapping.

    Args:
        root_dir (str): Path to the root directory of the dataset.

    Returns:
        datasets.ImageFolder: The adjusted dataset with merged labels.
    """
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


# --- Load Datasets ---
train_dataset = load_dataset(config.TRAIN_DIR)
val_dataset = load_dataset(config.VAL_DIR)

# --- Weighted Sampling for Class Imbalance ---
class_counts = [sum(1 for _, label in train_dataset.samples if label == i) for i in range(len(selected_classes))]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=config.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

print(len(selected_classes))
print(selected_classes)

# --- Model Preparation and Checkpoints ---
model = models.resnet50()

# Load Checkpoint for Resuming Training
start_epoch = 0
if config.RESUME_TRAINING:
    checkpoint_path = config.CHECKPOINT_PATH.format(config.LAST_EPOCH)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    fc_shape = checkpoint['model_state_dict']['fc.weight'].shape[0]
    model.fc = nn.Linear(model.fc.in_features, fc_shape)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"✅ Resume from {checkpoint_path}")
    start_epoch = config.LAST_EPOCH + 1
else:
    checkpoint = torch.load(config.PREVIOUS_CHECKPOINT, map_location=config.DEVICE)
    fc_shape = checkpoint['model_state_dict']['fc.weight'].shape[0]
    model.fc = nn.Linear(model.fc.in_features, fc_shape)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"📦 Loaded latest checkpoint: {config.PREVIOUS_CHECKPOINT}")

# Update Fully Connected Layer for Merged Classes
model.fc = nn.Linear(model.fc.in_features, len(selected_classes))
model = model.to(config.DEVICE)

# Freeze Layers except for Layer 3, Layer 4, and FC
for param in model.parameters():
    param.requires_grad = False
for block in [model.layer3, model.layer4, model.fc]:
    for param in block.parameters():
        param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
criterion = nn.CrossEntropyLoss()

# --- Training and Validation Loop ---
print("🚀 Starting Fine-tuning...")
for epoch in range(start_epoch, config.EPOCHS):
    model.train()
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}"):
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"✅ Epoch {epoch + 1} completed")

# --- Save Final Model and Configuration ---
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_path = os.path.join(config.MODEL_DIR, f"finetuned_model_{now}.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Fine-tuning complete. Model saved at: {model_path}")
