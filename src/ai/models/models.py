import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Ein Dummy-Bild mit der Größe (3, 224, 224) durch das Netzwerk schicken, um die Größe für fc1 zu berechnen
        self._get_conv_output_size()

        self.fc1 = nn.Linear(self.conv_output_size, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def _get_conv_output_size(self):
        # Erstelle ein Dummy-Bild der Größe (1, 3, 224, 224) und führe es durch das Netzwerk
        x = torch.zeros(1, 3, 224, 224)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # Berechne die Ausgabegröße nach den Convolutional- und Pooling-Schichten
        self.conv_output_size = x.numel()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
