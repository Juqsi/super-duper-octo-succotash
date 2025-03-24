import os
import shutil

# Verzeichnisse definieren
DATASET_DIR = "../../../dataset/plantnet_300K"
TRAIN_DIR = os.path.join(DATASET_DIR, "images_train")
VAL_DIR = os.path.join(DATASET_DIR, "images_val")
TEST_DIR = os.path.join(DATASET_DIR, "images_test")

# Mindestanzahl an Bildern pro Klasse
MIN_IMAGES = 5


def delete_small_folders(base_dir, min_images):
    """Löscht Unterordner, die weniger als `min_images` Dateien enthalten."""
    deleted_folders = []

    for class_folder in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_folder)

        if os.path.isdir(class_path):
            num_files = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])

            if num_files < min_images:
                shutil.rmtree(class_path)
                deleted_folders.append(class_folder)
                print(f"Gelöscht: {class_path} (nur {num_files} Dateien)")

    return deleted_folders


# Lösche zu kleine Ordner in images_train
deleted_classes = delete_small_folders(TRAIN_DIR, MIN_IMAGES)

# Lösche die gleichen Ordner in images_val und images_test
for dataset_dir in [VAL_DIR, TEST_DIR]:
    for class_folder in deleted_classes:
        class_path = os.path.join(dataset_dir, class_folder)

        if os.path.isdir(class_path):
            shutil.rmtree(class_path)
            print(f"Gelöscht: {class_path}")

print("Bereinigung abgeschlossen.")
