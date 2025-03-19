import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=15):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.1)  # Weniger Dropout in frühen CNN-Layern

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout4 = nn.Dropout(0.4)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 1024)
        self.dropout_fc = nn.Dropout(0.5)  # Hoher Wert in der FC-Schicht
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
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
