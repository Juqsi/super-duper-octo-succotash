import os

import torch

# === Verzeichnisse ===
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset/plantnet_300K"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dcheckpoints"))
PRETRAINED_MODEL_PATH = os.path.join(MODEL_DIR, 'resnet50.pth')
# Vorheriger Trainingsstand für Finetuning
PREVIOUS_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_116.pth')
RESUME_TRAINING = False
LAST_EPOCH = 49


# === Datasets ===
TRAIN_DIR = os.path.join(DATA_DIR, 'images_train')
VAL_DIR = os.path.join(DATA_DIR, 'images_val')
TEST_DIR = os.path.join(DATA_DIR, 'images_test')

# === Trainingsparameter ===
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
EPOCHS = 50
WEIGHT_DECAY = 0.0001
NUM_WORKERS = 12
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Checkpointing ===
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'finetune_checkpoint_epoch_{}.pth')

# === Merge-Konfiguration ===
MERGE_MAP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../helperscripts/merge_map.json'))
