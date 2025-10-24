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
        
        self.transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and scales between 0 and 1
        transforms.Normalize(
            mean=[0.4883, 0.4551, 0.4170],
            std=[0.2571, 0.2505, 0.2531]
        )
    ])


        self.train = DermaMNIST(split='train', download=True, size=self.batch_size, transform=self.transform)
        self.val = DermaMNIST(split='val', size=self.batch_size, transform=self.transform)
        self.test = DermaMNIST(split='test', size=self.batch_size, transform=self.transform)
        print(type(self.train))
 

    def Get_Info(self):
        info = INFO['dermamnist']

        n_channels = info['n_channels']
        n_classes = len(info['label'])
        n_samples = info['n_samples']



        return (n_channels, n_classes, n_samples)

    # def 
preprocessor = Preprocessing()