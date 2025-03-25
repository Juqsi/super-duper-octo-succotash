import os
import torch

# Verzeichnisse
DATA_DIR = '../../../dataset/plantnet_300K'
MODEL_DIR = '../../../models'
CHECKPOINT_DIR = '../../../checkpoints'

# Lokales Speichern des vortrainierten Modells
PRETRAINED_MODEL_PATH = "../../../models/resnet50.pth"

USE_MIXUP = True

MIXUP_ALPHA = 0.6  # Anfangswert
MIXUP_REDUCTION_EPOCH = 20

USE_CUTMIX = True
CUTMIX_PROB = 0.5  # 50 % Wahrscheinlichkeit für CutMix statt Mixup


# Hyperparameter
BATCH_SIZE = 128 # Reduziert auf 64 für besseres Generalisieren
LEARNING_RATE = 5e-5
EPOCHS = 120 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Datasets
TRAIN_DIR = os.path.join(DATA_DIR, 'images_train')
VAL_DIR = os.path.join(DATA_DIR, 'images_val')
TEST_DIR = os.path.join(DATA_DIR, 'images_test')

# Checkpoints
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_{}.pth')
RESUME_TRAINING = False # Wenn True, lädt er den letzten Checkpoint
LAST_EPOCH = 83  # Hier die letzte Epoche eintragen, ab der weitertrainiert wird

NUM_WORKERS = 12
WEIGHT_DECAY = 0.001
