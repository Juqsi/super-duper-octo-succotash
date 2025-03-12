import os
import torch

# Verzeichnisse
DATA_DIR = '../../../dataset/plantnet_300K'
MODEL_DIR = '../../../models'
CHECKPOINT_DIR = '../../../checkpoints'

# Hyperparameter
BATCH_SIZE =16
ACCUMULATION_STEPS = 2  # Simuliert größere Batch Size

LEARNING_RATE = 0.0001
EPOCHS = 6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Datasets
TRAIN_DIR = os.path.join(DATA_DIR, 'images_train')
VAL_DIR = os.path.join(DATA_DIR, 'images_val')
TEST_DIR = os.path.join(DATA_DIR, 'images_test')

# Checkpoints
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_{}.pth')
RESUME_TRAINING = False  # Wenn True, lädt er den letzten Checkpoint
LAST_EPOCH = 9  # Hier die letzte Epoche eintragen, ab der weitertrainiert wird

NUM_WORKERS = 4
WEIGHT_DECAY = 0.0001
WD = 0
LR =  0.0001


