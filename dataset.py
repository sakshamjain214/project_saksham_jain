# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import config

def get_dataloaders(use_subset=True, subset_size=20000):
    # Import resize variables from config.py
    data_transforms = transforms.Compose([
        transforms.Resize((config.resize_x, config.resize_y)),
        transforms.ToTensor()
    ])
    
    train_data = datasets.ImageFolder(root=config.train_dir, transform=data_transforms)
    valid_data = datasets.ImageFolder(root=config.valid_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(root=config.test_dir, transform=data_transforms)
    
    # Use the 20,000 subset you utilized in your final notebook run
    if use_subset:
        total_images = len(train_data)
        subset_indices = random.sample(range(total_images), min(subset_size, total_images))
        train_data = Subset(train_data, subset_indices)
    
    # Create the DataLoaders
    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader