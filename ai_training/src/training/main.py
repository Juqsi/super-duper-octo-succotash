import torch
print('GPU verfügbar:', torch.cuda.is_available())
print(torch.cuda.device_count(), 'GPUs gefunden')
print(torch.cuda.get_device_name(0))
