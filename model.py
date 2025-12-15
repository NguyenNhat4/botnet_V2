import torch
import torch.nn as nn

class Botnet1DCNN(nn.Module):
    def __init__(self, n_features, n_classes=3):
        super(Botnet1DCNN, self).__init__()

        # Less aggressive pooling, more feature extraction
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
            # NO POOLING HERE - preserve spatial information
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2)  # Only pool once
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Global pooling then classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x shape: [Batch, n_features]
        # Unsqueeze to create channel dimension: [Batch, 1, n_features]
        x = x.unsqueeze(1) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x

class BotnetClassifier(nn.Module):
    def __init__(self, base_model, n_features, image_size=None, n_classes=3):
        # image_size is deprecated but kept for compatibility if needed
        super().__init__()
        self.model = Botnet1DCNN(n_features=n_features, n_classes=n_classes)

    def forward(self, x):
        return self.model(x)
