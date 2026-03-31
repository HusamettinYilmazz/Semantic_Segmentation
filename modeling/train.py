import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW

import albumentations as A
from albumentations.pytorch import ToTensorV2

from unet_model import UNet
from utils.dataset import VOCDataset

ROOT = ""

def train(config, checkpoint_path=None):

    dataset_path = os.path.join(ROOT, config.data['dataset_path'])
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.ColorJitter(brightness=0.2),
            A.RandomBrightnessContrast(),
            A.GaussNoise()
        ], p=0.5),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=0.05),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    train_val_dataset_path = os.path.join(dataset_path, config.data['train_dataset_path'])
    train_dataset = VOCDataset(data_path= train_val_dataset_path,
                               data_type="train",
                               transform=train_transform)
    
    val_dataset = VOCDataset(data_path= train_val_dataset_path,
                               data_type="val",
                               transform=val_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.training['batch_size'], shuffle= True, num_workers=4, pin_memory= True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.training['batch_size'], shuffle= True, num_workers=4, pin_memory= True)

    model = UNet(out_features=config.model['num_classes'])
    model = model.to(device)

    optimizer = AdamW(params= model.parameters(), lr=float(config.training['learning_rate']), weight_decay=float(config.training['weight_decay']))
    

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(1, config.training['num_epochs']+1):

        ## train one epoch
        
        ## evaluate the model
        
        ...

"""
1- Data path
2- Train/Val data and their augmentation as dataset instance
3- DataLoader
4- Model
5- Optimizer
6- Loss function
7- Training loop

"""