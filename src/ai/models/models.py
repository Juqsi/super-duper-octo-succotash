import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Berechne die Größe der Ausgabedaten nach den Convolutional-Schichten
        self.conv_output_size = self._get_conv_output_size()

        self.fc1 = nn.Linear(self.conv_output_size, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def _get_conv_output_size(self):
        # Dummy-Tensor für die Berechnung der Ausgabegröße nach den Convolutional-Schichten
        x = torch.zeros(1, 3, 224, 224)  # Ein Dummy-Bild der Größe (1, 3, 224, 224)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

