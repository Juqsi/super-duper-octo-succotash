import time
from datetime import datetime
import torch
import os
import json
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import config
from src.ai.models.models import MyModel
from utils import save_checkpoint, load_checkpoint
import glob
import math


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

            # Überprüfe NaN in Eingabedaten
            if torch.isnan(inputs).any() or torch.isnan(labels).any():
                print("NaN detected in inputs or labels during validation!")
                return float('nan'), 0.0

            outputs = model(inputs)

            # Überprüfe NaN in Modellvorhersagen
            if torch.isnan(outputs).any():
                print("NaN detected in model output during validation!")
                return float('nan'), 0.0

            loss = criterion(outputs, labels)

            # Überprüfe NaN im Verlust
            if torch.isnan(loss).any():
                print("NaN detected in loss during validation!")
                return float('nan'), 0.0

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    sval_loss = val_loss / len(val_loader)
    val_acc = correct / total
    return sval_loss, val_acc


# Funktion zum Berechnen der Trainingszeit
def get_time_elapsed(start_time):
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    return f"{minutes}m {seconds}s"


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
#selected_classes = os.listdir(config.TRAIN_DIR)  # Z.B. nur die ersten 10 Klassen
selected_classes = ['1356126','1363128','1356022','1357330','1355978','1363740','1364172','1355937','1361656','1363021','1385937','1356421','1358094','1384485','1393614']
print("Selected Classes:", selected_classes)

# Lade die Datasets
train_dataset = load_filtered_dataset(config.TRAIN_DIR, selected_classes)
val_dataset = load_filtered_dataset(config.VAL_DIR, selected_classes)
test_dataset = load_filtered_dataset(config.TEST_DIR, selected_classes)

print(f"Anzahl der Trainingsbeispiele: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

# 2. Modell initialisieren
model = MyModel(num_classes=len(selected_classes)).to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3) # cos funktion

# 3. Checkpoint laden, falls Modell fortgesetzt werden soll
start_epoch = 0

if config.RESUME_TRAINING:
    checkpoint_path = config.CHECKPOINT_PATH.format(config.LAST_EPOCH)
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path, config.DEVICE)
    else:
        print(f"Checkpoint {checkpoint_path} nicht gefunden. Starte bei Epoche 0")

print(f"Starte Training ab Epoche {start_epoch}...")

# Startzeit des Trainings
training_start_time = time.time()

# 4. Training
epoch_times = []
for epoch in range(start_epoch, config.EPOCHS):
    epoch_start_time = time.time()  # Startzeit für die Epoche

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

        # Überprüfe NaN in Eingabedaten
        if torch.isnan(inputs).any() or torch.isnan(labels).any():
            print("NaN detected in inputs or labels during training!")
            break

        optimizer.zero_grad()
        outputs = model(inputs)

        # Überprüfe NaN in Modellvorhersagen
        if torch.isnan(outputs).any():
            print("NaN detected in model output during training!")
            break

        loss = criterion(outputs, labels)

        # Überprüfe NaN im Verlust
        if torch.isnan(loss).any():
            print("NaN detected in loss during training!")
            break

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Zeige die Statistiken der aktuellen Epoche
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f"Epoch [{epoch + 1}/{config.EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Validierung nach jeder Epoche
    sval_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)

    # Überprüfe NaN für float-Werte
    if math.isnan(sval_loss) or math.isnan(val_acc):
        print("NaN detected during validation. Aborting training.")
        break

    print(f"Validation Loss: {sval_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    scheduler.step(sval_loss)
    print(f"Neue Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    # Speichern des Modells nach jeder Epoche
    save_checkpoint(model, optimizer, epoch, config.CHECKPOINT_PATH.format(epoch))
    clean_old_checkpoints(config.CHECKPOINT_DIR)

    epoch_end_time = time.time()  # Endzeit für die Epoche
    epoch_times.append(get_time_elapsed(epoch_start_time))  # Zeit für diese Epoche speichern

# Gesamtzeit des Trainings
training_end_time = time.time()
total_training_time = get_time_elapsed(training_start_time)

# Modell am Ende speichern
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_filename = os.path.join(config.MODEL_DIR, f'final_model_{timestamp}.pth')
config_filename = os.path.join(config.MODEL_DIR, f'config_{timestamp}.json')

torch.save(model.state_dict(), model_filename)

# Konfigurationsdatei speichern
config_data = {
    'BATCH_SIZE': config.BATCH_SIZE,
    'LEARNING_RATE': config.LEARNING_RATE,
    'EPOCHS': config.EPOCHS,
    'DEVICE': config.DEVICE,
    'TRAIN_DIR': config.TRAIN_DIR,
    'VAL_DIR': config.VAL_DIR,
    'TEST_DIR': config.TEST_DIR,
    'CHECKPOINT_PATH': config.CHECKPOINT_PATH,
    'RESUME_TRAINING': config.RESUME_TRAINING,
    'LAST_EPOCH': config.LAST_EPOCH,
    'NUM_WORKERS': config.NUM_WORKERS,
    'WEIGHT_DECAY': config.WEIGHT_DECAY,
    'TOTAL_TRAINING_TIME': total_training_time,
    'EPOCH_TIMES': epoch_times,  # Zeiten der einzelnen Epochen
    'NUM_CLASSES': len(selected_classes),  # Anzahl der Klassen
    'CLASSES': selected_classes  # Die Klassennamen
}

# Speichern der Konfigurationsdatei als JSON
with open(config_filename, 'w') as f:
    json.dump(config_data, f, indent=4)

print(f"Modelle gespeichert als: {model_filename}")
print(f"Konfiguration gespeichert als: {config_filename}")
