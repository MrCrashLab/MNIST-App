import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), padding=1)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=1)    #28x28
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))                                             #14x14
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3))               #12x12
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))                                             #6x6
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3))               #4x4
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))                                             #2x2
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=256, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x