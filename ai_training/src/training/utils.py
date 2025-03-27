import torch


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """
    Speichert den aktuellen Zustand des Modells und Optimizers als Checkpoint.

    Diese Funktion speichert den Zustand des Modells, des Optimizers und die aktuelle Epoche
    in einem Dictionary und schreibt dieses Dictionary in eine Datei, die durch 'checkpoint_path'
    spezifiziert ist.

    Args:
        model (torch.nn.Module): Das zu speichernde Modell.
        optimizer (torch.optim.Optimizer): Der Optimizer, dessen Zustand ebenfalls gespeichert wird.
        epoch (int): Die aktuelle Epoche, die im Checkpoint vermerkt wird.
        checkpoint_path (str): Der Dateipfad, an dem der Checkpoint gespeichert wird.

    Returns:
        None

    Side Effects:
        - Speichert einen Checkpoint auf der Festplatte.
        - Gibt eine Bestätigung auf der Konsole aus.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint für Epoche {epoch} gespeichert.")


def load_checkpoint(model, optimizer, path, device):
    """
    Lädt einen gespeicherten Checkpoint und stellt den Zustand von Modell und Optimizer wieder her.

    Diese Funktion lädt einen Checkpoint von dem angegebenen Pfad, stellt den Zustand des Modells und
    des Optimizers wieder her und gibt die geladene Epoche zurück.

    Args:
        model (torch.nn.Module): Das Modell, in das der gespeicherte Zustand geladen wird.
        optimizer (torch.optim.Optimizer): Der Optimizer, in den der gespeicherte Zustand geladen wird.
        path (str): Der Pfad zur Checkpoint-Datei.
        device (torch.device or str): Das Gerät, auf das die geladenen Daten gemappt werden sollen.

    Returns:
        tuple: Ein Tupel (model, optimizer, epoch) bestehend aus dem Modell mit geladenem Zustand,
               dem Optimizer mit geladenem Zustand und der zuletzt gespeicherten Epoche.

    Side Effects:
        - Gibt Statusmeldungen auf der Konsole aus.
    """
    print(f"Lade Checkpoint von {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=True)  # Sicherer laden
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint geladen (Epoche {epoch})")
    return model, optimizer, epoch
