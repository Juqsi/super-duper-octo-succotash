"""
Dieses Skript dient zur Bereinigung von Datensatzverzeichnissen, indem Klassenordner, die weniger als eine bestimmte
Mindestanzahl an Bildern enthalten, gelöscht werden.
Es werden drei Verzeichnisse verarbeitet: images_train, images_val und images_test des PlantNet 300K Datensatzes.
Zunächst werden zu kleine Ordner im Trainingsdatensatz gelöscht. Anschließend werden die gleichen Ordner in den
Validierungs- und Testdatensätzen entfernt.
"""

import os
import shutil

# Verzeichnisse definieren

TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset/plantnet_300K/images_train"))
if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR, exist_ok=True)
VAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset/plantnet_300K/images_val"))
if not os.path.exists(VAL_DIR):
    os.makedirs(VAL_DIR, exist_ok=True)
TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset/plantnet_300K/images_test"))
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR, exist_ok=True)

# Mindestanzahl an Bildern pro Klasse
MIN_IMAGES = 5


def delete_small_folders(base_dir, min_images):
    """
    Löscht Unterordner im angegebenen Basisverzeichnis, die weniger als 'min_images' Dateien enthalten.

    Durchläuft alle Unterordner in 'base_dir' und entfernt diejenigen, in denen die Anzahl der Dateien kleiner
    als 'min_images' ist. Dabei werden nur Ordner berücksichtigt, die tatsächlich existieren und Dateien enthalten.

    Args:
        base_dir (str): Das Verzeichnis, in dem nach Unterordnern gesucht wird.
        min_images (int): Mindestanzahl an Dateien, die ein Unterordner enthalten muss, um nicht gelöscht zu werden.

    Returns:
        list: Eine Liste der Namen der gelöschten Unterordner.
    """
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
