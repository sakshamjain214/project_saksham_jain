# model.py
import torch.nn as nn
import config

class DeepfakeDetectorCNN(nn.Module):
    def __init__(self):
        super(DeepfakeDetectorCNN, self).__init__()

        # Convolutional Blocks
        self.Conv1 = nn.Conv2d(in_channels=config.input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16) 
        
        self.Conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.Conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.Conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        # Fully Connected Block
        self.Lin1 = nn.Linear(in_features=14*14*128, out_features=256)
        self.drop1 = nn.Dropout(0.5) 
        
        self.Lin2 = nn.Linear(in_features=256, out_features=32)
        self.drop2 = nn.Dropout(0.5)
        
        self.Lin3 = nn.Linear(in_features=32, out_features=2) 
    
    def forward(self, x):
        # Block 1
        x = self.Conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.Conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 3
        x = self.Conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 4
        x = self.Conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = self.flatten(x)
        
        # Linear Block
        x = self.Lin1(x)
        x = self.relu(x)
        x = self.drop1(x)
        
        x = self.Lin2(x)
        x = self.relu(x)
        x = self.drop2(x)
        
        x = self.Lin3(x)
        return x