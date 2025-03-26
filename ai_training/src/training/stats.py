import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import pandas as pd


def save_confusion_matrix(model, val_loader, selected_classes, device, epoch, output_dir):
    """
    Speichert die normalisierte Confusion Matrix als Bild und CSV-Datei und gibt die häufigsten Fehlklassifikationen aus.

    Diese Funktion führt folgende Schritte durch:
      1. Berechnet Vorhersagen des Modells auf dem Validierungs-Dataset.
      2. Erzeugt eine normalisierte Confusion Matrix.
      3. Visualisiert die Matrix mit Hilfe von Matplotlib und speichert das Diagramm als PNG-Datei.
      4. Speichert die normalisierte Matrix als CSV-Datei.
      5. Findet und gibt die Top-N (standardmäßig 30) häufigsten Fehlklassifikationen (außerhalb der Diagonale) aus.

    Args:
        model (torch.nn.Module): Das zu evaluierende Modell.
        val_loader (torch.utils.data.DataLoader): DataLoader für den Validierungsdatensatz.
        selected_classes (list): Liste der Klassennamen.
        device (torch.device or str): Das Gerät, auf dem die Berechnungen durchgeführt werden (z. B. "cpu" oder "cuda").
        epoch (int): Aktuelle Epoche, die zur Kennzeichnung der gespeicherten Dateien verwendet wird.
        output_dir (str): Verzeichnis, in dem die Bild- und CSV-Dateien gespeichert werden.

    Returns:
        None
    """
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Erstellen und speichern der Confusion Matrix Visualisierung
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=selected_classes)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=90, values_format=".2f")
    plt.title(f"Normalized Confusion Matrix (Epoch {epoch})")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_epoch_{epoch}.png"))

    # Speichern der normalisierten Confusion Matrix als CSV-Datei
    df_cm = pd.DataFrame(cm_normalized, index=selected_classes, columns=selected_classes)
    df_cm.to_csv(os.path.join(output_dir, f"confusion_matrix_normalized_epoch_{epoch}.csv"))

    # Top-N Fehlklassifikationen ermitteln (außerhalb der Diagonale)
    N = 30  # Anzahl der häufigsten Fehler
    error_matrix = cm_normalized.copy()
    np.fill_diagonal(error_matrix, 0)  # Diagonale ignorieren = keine Treffer
    flat_indices = np.argsort(error_matrix.ravel())[::-1][:N]  # Top-N Fehler
    top_misclassifications = [np.unravel_index(idx, error_matrix.shape) for idx in flat_indices]

    print(f"\nTop-{N} häufigste Verwechslungen in Epoche {epoch}:")
    for i, j in top_misclassifications:
        true_label = selected_classes[i]
        predicted_label = selected_classes[j]
        confusion_value = error_matrix[i, j]
        print(f"  {true_label} → {predicted_label}: {confusion_value:.2f}")

    # Diagramm schließen
    plt.close()


def plot_training_progress(train_acc, val_acc, top5_acc, output_dir):
    """
    Zeichnet und speichert den Verlauf der Trainings- und Validierungsgenauigkeiten über die Epochen.

    Diese Funktion erstellt einen Plot, der die Trainingsgenauigkeit, die Validierungsgenauigkeit (Top-1)
    sowie die Validierungsgenauigkeit (Top-5) im Verlauf der Epochen anzeigt. Der resultierende Plot
    wird als PNG-Datei im angegebenen Ausgabeverzeichnis gespeichert.

    Args:
        train_acc (list): Liste der Trainingsgenauigkeiten pro Epoche.
        val_acc (list): Liste der Validierungsgenauigkeiten (Top-1) pro Epoche.
        top5_acc (list): Liste der Validierungsgenauigkeiten (Top-5) pro Epoche.
        output_dir (str): Verzeichnis, in dem der Plot gespeichert werden soll.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy (Top-1)')
    plt.plot(epochs, top5_acc, label='Validation Accuracy (Top-5)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_accuracy_plot.png"))
    plt.close()
