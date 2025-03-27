import os

import torch

# Verzeichnisse
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset/plantnet_300K"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints"))

# Lokales Speichern des vortrainierten Modells
PRETRAINED_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/resenet50.pth"))

USE_MIXUP = False

MIXUP_ALPHA = 0.6
MIXUP_REDUCTION_EPOCH = 20

USE_CUTMIX = False
CUTMIX_PROB = 0.5


# Hyperparameter
BATCH_SIZE = 128  # Reduziert auf 64 für besseres Generalisieren
LEARNING_RATE = 5e-5
EPOCHS = 120
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Datasets
TRAIN_DIR = os.path.join(DATA_DIR, 'images_train')
VAL_DIR = os.path.join(DATA_DIR, 'images_val')
TEST_DIR = os.path.join(DATA_DIR, 'images_test')

# Checkpoints
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_{}.pth')
RESUME_TRAINING = True
LAST_EPOCH = 118

NUM_WORKERS = 12
WEIGHT_DECAY = 0.001
