import os
import sys
ROOT = os.getcwd()
sys.path.append(ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW, lr_scheduler
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from modeling.unet import UNet
from utils.dataset import VOCDataset
from utils import lr_vs_epoch, save_checkpoint, Logger
from utils.eval import compute_confusion_matrix, compute_iou_per_class
from utils.eval import compute_per_class_accuracy, plot_confusion_matrix
from utils import load_config

def train_an_epoch(epoch, data_loader, device, model, optimizer, loss_func, scaler, logger):
    total_loss = 0.0
    model.train()
    for batch_idx, (imgs, masks) in enumerate(data_loader):
        imgs, masks = imgs.to(device), masks.to(device).long()
        optimizer.zero_grad()
        outputs = model(imgs)

        loss = loss_func(outputs, masks)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"TRAIN: Epoch:{epoch} at Batch:{batch_idx}/{len(data_loader)} Loss:{loss.item():.3f}")
    
    avg_loss = total_loss/ len(data_loader)
    logger.info(f"Epoch:{epoch} average train Loss:{avg_loss:.3f}")
    return avg_loss

def validate_model(epoch, data_loader, device, model, loss_func, class_names, logger, save_dir=None):
    total_loss = 0.0
    total_cm = None

    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(data_loader):
            imgs, masks = imgs.to(device), masks.to(device).long()

            outputs = model(imgs)
            loss = loss_func(outputs, masks)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)

            cm = compute_confusion_matrix(masks, preds, class_names, ignore_index=255)

            total_cm = cm if total_cm is None else total_cm + cm

    plot_confusion_matrix(total_cm, class_names, save_path=save_dir)
    iou_per_class = compute_iou_per_class(total_cm)
    acc_per_class = compute_per_class_accuracy(total_cm)

    metrics = {
        "avg_loss" : total_loss / len(data_loader),
        "iou_per_class" : iou_per_class,
        "acc_per_class" : acc_per_class,
    }

    logger.info(f"Epoch:{epoch} validation metrics")
    logger.info(metrics)
    return metrics

def train(config, checkpoint_path=None):

    dataset_path = os.path.join(ROOT, config.data['dataset_path'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose([
        A.Resize(512, 512),
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
        A.Resize(512, 512),
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
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.training['batch_size'], shuffle= True, pin_memory= True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.training['batch_size'], shuffle= True, pin_memory= True)

    model = UNet(out_classes=config.model['num_classes'])
    model = model.to(device)

    optimizer = AdamW(params= model.parameters(), lr=float(config.training['learning_rate']), weight_decay=float(config.training['weight_decay']))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=2)

    loss_func = nn.CrossEntropyLoss(ignore_index=255)
    scaler = GradScaler()

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        starting_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.param_groups[0]['lr'] = checkpoint['learning_rate']

    else:
        starting_epoch = 1

    save_dir = os.path.join(ROOT, config.data['output_path'], config.experiment['name'], config.experiment['version'])
    os.makedirs(save_dir, exist_ok=True)

    logger = Logger(save_dir)
    logger.info(f"Starting the experiment: {config.experiment['name']} {config.experiment['version']}")
    logger.info(f"Using device: {device}")
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    lrs = []
    logger.info(f"Starting training from epoch: {starting_epoch}")
    
    for epoch in range(1, config.training['num_epochs']+1):
        logger.info(f"Epoch: {epoch}/{config.training['num_epochs']}")
        
        save_file = os.path.join(save_dir, f'epoch{epoch}_conf_matrix.png')
        
        train_avg_loss = train_an_epoch(epoch, train_loader, device, model, optimizer, loss_func, scaler, logger)
        
        val_metrics = validate_model(epoch, val_loader, device, model, loss_func, config.model['class_labels'], logger, save_file)
        

        logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        scheduler.step(val_metrics['avg_loss'])
        
        cur_lr = optimizer.param_groups[0]['lr']
        lrs.append(cur_lr)
        
        save_checkpoint(epoch, model, optimizer, cur_lr, val_metrics['acc_per_class'], config, train_transform, val_transform, save_dir)
    logger.info(f"Training completed successfully")

    lr_vs_epoch(config.training['num_epochs']-starting_epoch+1, lrs, save_dir)


"""
Pritiorize Doing (NO NEED TO UNDERSTAND THE HALL TORCH LIBRARY AND ITS ALL INTERNAL SHIT TO MOVE ON)

1- Data path
2- Train/Val data and their augmentation as dataset instance
3- DataLoader
4- Model
5- Optimizer
6- Loss function
7- Training loop

"""


if __name__ == "__main__":
    config = load_config(os.path.join(ROOT, "config/unet_config.yml"))
    train(config)

"""

what to do next:
    1. go to cloud because Cuda is out of memory :)
    
"""