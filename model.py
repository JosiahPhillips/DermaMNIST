import torch
import numpy as np
import torch.nn as nn
from preprocessing import Preprocessor


class Model(nn.Module):
    def __init__(self, input_size, output_size, criterion, optimizer, kernel_size):
        
        self.input_size = input_size
        self.output_size = output_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.kernel_size = kernel_size

        # Define the model's Layers
        self.layer1 = nn.Conv2d(self.input_size, (32,32), self.kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(self.kernel_size)
        self.layer2 = nn.Conv2d((32,32), (16,16), self.kernel_size, padding=1)
        self.layer3 = nn.Conv2d((16,16), (8,8), self.kernel_size, padding=1)
        self.flatten() = nn.Flatten()
        self.layer4 = nn.Linear(64, output_size)

        return 0
    def forward():
        return 0
    def predict():
        return
