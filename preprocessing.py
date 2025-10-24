import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import DermaMNIST, INFO
import numpy as np
from pathlib import Path


class Preprocessing:

    def __init__(self):
        self.results_dir = Path("./results")
        self.results_dir.mkdir(exist_ok=True)

        self.batch_size = 64
        self.n_channels, self.n_classes, self.n_samples = self.Get_Info()
    
        # print("Number of Channels: ", self.n_channels)
        # print("Number of Classes: ", self.n_classes)
        # print("Number of Samples: ", self.n_samples)
    
        self.train = DermaMNIST(split='train', download=True, size=self.batch_size, transform=transforms)
        self.val = DermaMNIST(split='val', size=self.batch_size, transform=transforms)
        self.test = DermaMNIST(split='test', size=self.batch_size, transform=transforms)
 

    def Get_Info(self):
        info = INFO['dermamnist']

        n_channels = info['n_channels']
        n_classes = len(info['label'])
        n_samples = info['n_samples']



        return (n_channels, n_classes, n_samples)


preprocessor = Preprocessing()