import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Simple CNN for MNIST.
    Input:  (N, 1, 28, 28)
    Output: (N, 10) logits
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)   # -> (N,32,26,26)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)  # -> (N,64,24,24)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)                      # halves spatial dims

        # After conv1: 28->26, pool -> 13
        # After conv2: 13->11, pool -> 5
        # So final feature map: (N,64,5,5) = 64*5*5
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)   # (N, 64*5*5)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)           # logits
        return x
