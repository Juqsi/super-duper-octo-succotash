import torch

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Speichert den Modell-Checkpoint nach jeder Epoche."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint für Epoche {epoch} gespeichert.")

def load_checkpoint(model, optimizer, path, device):
    print(f"Lade Checkpoint von {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=True)  # Sicherer laden
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint geladen (Epoche {epoch})")
    return model, optimizer, epoch

