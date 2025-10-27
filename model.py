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
        self.layer_kernel_size = kernel_size
        self.pool_kernel_size = 2
        self.total_loss = 0


        # Define the model's Layers
        self.layer1 = nn.Conv2d(self.input_size, 16, self.layer_kernel_size, padding=1)
        self.layer2 = nn.Conv2d(16, 32, self.layer_kernel_size, padding=1)
        self.layer3 = nn.Conv2d(32, 64, self.layer_kernel_size, padding=1)
        self.layer4 = nn.Linear(64, output_size)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(self.pool_kernel_size)
        

       
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x
    
    def one_epoch(self, x, y):
        self.optimizer.zero_grad() # clear old gradients
        outputs = self.forward(x) # run the model
        print(outputs.shape)
        loss = self.criterion(outputs, y) # calcualte loss
        loss.backward() #compute the gradients
        self.optimizer.step() # update the weights
        return loss.item() # numerical loss
    def predict():
        return
