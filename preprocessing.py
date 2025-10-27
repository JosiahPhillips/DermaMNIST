import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import DermaMNIST, INFO
import numpy as np
from pathlib import Path


class Preprocessor:

    def __init__(self, std_vals=[0.1357, 0.1572, 0.1751], mean_vals=[0.7632, 0.5380, 0.5615]):
        self.results_dir = Path("./results")
        self.results_dir.mkdir(exist_ok=True)

        self.n_channels, self.n_classes, self.n_samples = self.Get_Info()
    
        # print("Number of Channels: ", self.n_channels)
        # print("Number of Classes: ", self.n_classes)
        # print("Number of Samples: ", self.n_samples)
        
        self.transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and scales between 0 and 1
        transforms.Normalize(
            mean=mean_vals,
            std=std_vals
        )
    ])

        self.train = DermaMNIST(split='train', download=True, size=64, transform=self.transform)
        self.val = DermaMNIST(split='val', size=64, transform=self.transform)
        self.test = DermaMNIST(split='test', size=64, transform=self.transform)
 

    def Get_Info(self):
        info = INFO['dermamnist']

        n_channels = info['n_channels']
        n_classes = len(info['label'])
        n_samples = info['n_samples']

        return (n_channels, n_classes, n_samples)


    def Create_Loaders(self, batch_size):
        # Batch shape (Batch_size, channels, H, W)
        self.train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(self.val, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=False)
        # imgs, _ = next(iter(self.train_loader))
        # mean = imgs.mean(dim=[0, 2, 3])
        # std = imgs.std(dim=[0, 2, 3])

        # print("Mean after transform:", mean)
        # print("Std after transform:", std)
        return self.train_loader, self.val_loader, self.test_loader
    def Calc_STD(self):
        train_imgs = torch.tensor(self.train.imgs, dtype=torch.float32) / 255.0       
        std_vals = train_imgs.std(dim=(0,1,2))
        return std_vals
    
    def Calc_Mean(self):
        train_imgs = torch.tensor(self.train.imgs, dtype=torch.float32) / 255.0       
        mean_vals = train_imgs.mean(dim=(0,1,2))
        return mean_vals
    
# preprocessor = Preprocessor()
# preprocessor.Create_Loaders(32)
# std_vals = preprocessor.Calc_STD()
# mean_vals = preprocessor.Calc_Mean()

# # print(std_vals)
# # print(mean_vals)