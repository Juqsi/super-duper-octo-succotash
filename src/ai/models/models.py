import torch
import torch.nn as nn

class MyModel(nn.Module):
        def __init__(self, num_classes=15):
            super(MyModel, self).__init__()

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)

            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)

            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)

            self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(512)

            self.pool = nn.MaxPool2d(2, 2)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Automatische Anpassung

            self.fc1 = nn.Linear(512, 1024)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(1024, num_classes)

        def forward(self, x):
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool(torch.relu(self.bn3(self.conv3(x))))
            x = self.pool(torch.relu(self.bn4(self.conv4(x))))

            x = self.adaptive_pool(x)  # Automatische Reduzierung der Feature-Map
            x = torch.flatten(x, 1)

            x = torch.relu(self.fc1(x))
            x = self.dropout(x)  # Dropout für mehr Robustheit
            x = self.fc2(x)

            return x


