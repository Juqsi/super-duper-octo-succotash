import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from src.ai.models.models import MyModel
from utils import save_checkpoint, load_checkpoint
import config
import glob

def clean_old_checkpoints(save_dir, keep=3):
    checkpoints = sorted(glob.glob(f"{save_dir}/*.pth"), key=os.path.getmtime)
    while len(checkpoints) > keep:
        os.remove(checkpoints[0])
        checkpoints.pop(0)
        print(f"Checkpoint {checkpoints[0]} gelöscht")


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Überprüfe, ob die Dimensionen der Vorhersagen und Labels übereinstimmen
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    sval_loss = val_loss / len(val_loader)
    val_acc = correct / total
    return sval_loss, val_acc


# 1. Datasets und Dataloader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Lade gefilterte Datasets
def load_filtered_dataset(root_dir, selected_classes):
    dataset = datasets.ImageFolder(root_dir, transform=transform)

    # Stelle sicher, dass die ausgewählten Klassen im Dataset vorhanden sind
    valid_classes = [cls for cls in selected_classes if cls in dataset.classes]

    # Filtere die Samples basierend auf den validen Klassen
    dataset.samples = [s for s in dataset.samples if dataset.classes[s[1]] in valid_classes]

    # Filtere die `class_to_idx` basierend auf den validen Klassen
    dataset.class_to_idx = {class_: idx for idx, class_ in enumerate(valid_classes)}
    dataset.classes = valid_classes

    return dataset


# Wähle Klassen
selected_classes = os.listdir(config.TRAIN_DIR)[:10]  # Z.B. nur die ersten 10 Klassen
print(selected_classes)

# Lade die Datasets
train_dataset = load_filtered_dataset(config.TRAIN_DIR, selected_classes)
val_dataset = load_filtered_dataset(config.VAL_DIR, selected_classes)
test_dataset = load_filtered_dataset(config.TEST_DIR, selected_classes)

print(f"Anzahl der Trainingsbeispiele: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,num_workers=config.NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,num_workers=config.NUM_WORKERS)

# 2. Modell initialisieren
model = MyModel(num_classes=len(selected_classes)).to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# 3. Checkpoint laden, falls Modell fortgesetzt werden soll
start_epoch = 0

if config.RESUME_TRAINING:
    checkpoint_path = config.CHECKPOINT_PATH.format(config.LAST_EPOCH)
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, config.DEVICE)
    else:
        print(f"Checkpoint {checkpoint_path} nicht gefunden. Starte bei Epoche 0")

print(f"Starte Training ab Epoche {start_epoch}...")


# 4. Training
for epoch in range(start_epoch, config.EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Zeige die Statistiken der aktuellen Epoche
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Speichern des Modells nach jeder Epoche
    sval_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
    save_checkpoint(model, optimizer, epoch, config.CHECKPOINT_PATH.format(epoch))
    clean_old_checkpoints(config.CHECKPOINT_DIR)

# Modell am Ende speichern
torch.save(model.state_dict(), config.MODEL_DIR + '/final_model.pth')
print("Training abgeschlossen und Modell gespeichert!")
