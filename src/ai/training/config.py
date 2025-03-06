import os
import torch

# Verzeichnisse
DATA_DIR = '../../../dataset/plantnet_300K'
MODEL_DIR = '../../../models'
CHECKPOINT_DIR = '../../../checkpoints'

# Hyperparameter
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Datasets
TRAIN_DIR = os.path.join(DATA_DIR, 'images_train')
VAL_DIR = os.path.join(DATA_DIR, 'images_val')
TEST_DIR = os.path.join(DATA_DIR, 'images_test')

# Checkpoints
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_{}.pth')
RESUME_TRAINING = False  # Wenn True, lädt er den letzten Checkpoint
LAST_EPOCH = 0  # Hier die letzte Epoche eintragen, ab der weitertrainiert wird

NUM_WORKERS = 8
WEIGHT_DECAY = 1e-3
