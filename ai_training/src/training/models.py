import torch
import torch.nn as nn


class MyModel(nn.Module):
    """
    Convolutional Neural Network (CNN) für Bildklassifikation.

    Dieses Modell besteht aus mehreren Convolutional-Blöcken mit Batch-Normalisierung,
    ReLU-Aktivierung, Dropout und Pooling-Schichten. Anschließend werden die extrahierten
    Merkmale durch voll verbundene Schichten geleitet, um die finale Klassifizierung durchzuführen.

    Args:
        num_classes (int): Anzahl der Ausgabeklassen für die Klassifikation. Standardwert ist 15.
    """

    def __init__(self, num_classes=15):
        """
        Initialisiert das MyModel und definiert die Netzwerkarchitektur.

        Erstellt die Convolutional-, Batch-Normalisierungs-, Dropout-, Pooling- und Fully-Connected-Schichten
        des Netzwerks.

        Args:
            num_classes (int): Anzahl der Ausgabeklassen. Standard ist 15.
        """
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.05)  # Weniger Dropout in frühen CNN-Layern

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout4 = nn.Dropout(0.4)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 1024)
        self.dropout_fc = nn.Dropout(0.6)  # Hoher Wert in der FC-Schicht
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        """
        Führt einen Forward-Pass des Modells durch.

        Verarbeitet die Eingabe durch eine Abfolge von Convolutional-Blöcken,
        die jeweils aus Convolution, Batch-Normalisierung, ReLU, Pooling und Dropout bestehen.
        Anschließend wird mittels adaptivem Pooling der Feature-Vektor reduziert und durch
        voll verbundene Schichten geleitet, um die finale Klassifizierung zu berechnen.

        Args:
            x (torch.Tensor): Eingabetensor der Form (N, C, H, W), wobei N die Batch-Größe darstellt.

        Returns:
            torch.Tensor: Ausgabe des Netzwerks (Logits) für jede Klasse.
        """
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)  # Dropout nach dem ersten Block

        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)  # Dropout vor der letzten Schicht
        x = self.fc2(x)

        return x
