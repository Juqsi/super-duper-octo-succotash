import os
import torch

# Verzeichnisse
DATA_DIR = '../../../dataset/plantnet_300K'
MODEL_DIR = '../../../models'
CHECKPOINT_DIR = '../../../checkpoints'

# Lokales Speichern des vortrainierten Modells
PRETRAINED_MODEL_PATH = "../../../models/resnet50.pth"

USE_MIXUP = True

MIXUP_ALPHA = 0.4  # Anfangswert
MIXUP_REDUCTION_EPOCH = 10

# Hyperparameter
BATCH_SIZE = 64 # Reduziert auf 64 für besseres Generalisieren
LEARNING_RATE = 0.0003
EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Datasets
TRAIN_DIR = os.path.join(DATA_DIR, 'images_train')
VAL_DIR = os.path.join(DATA_DIR, 'images_val')
TEST_DIR = os.path.join(DATA_DIR, 'images_test')

# Checkpoints
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_{}.pth')
RESUME_TRAINING = False  # Wenn True, lädt er den letzten Checkpoint
LAST_EPOCH = 14  # Hier die letzte Epoche eintragen, ab der weitertrainiert wird

NUM_WORKERS = 8
WEIGHT_DECAY = 0.001

# Dynamische Lernraten-Reduktion
LR_DECAY_EPOCH = 15  # Ab welcher Epoche die LR reduziert wird
LR_DECAY_FACTOR = 0.1  # Faktor, um den die Lernrate reduziert wird (z. B. 0.1 = 10x kleiner)
