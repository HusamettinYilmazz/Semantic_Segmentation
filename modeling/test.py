import os
import sys
ROOT = os.getcwd()
sys.path.append(ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from modeling.unet import UNet
from modeling.train import validate_model

from utils.dataset import VOCDataset
from utils import load_config, Logger





def test_model(config, model_path, loss_func, test_transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check_point = torch.load(model_path, map_location= device)

    test_dataset_path = os.path.join(ROOT, config.data['dataset_path'], config.data['train_dataset_path'])

    model = UNet(out_classes=config.model['num_classes'])
    model.load_state_dict(check_point['model_state_dict'])
    model.to(device)

    test_dataset = VOCDataset(data_path= test_dataset_path,
                               data_type="val",
                               transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.training['batch_size'], shuffle= True, pin_memory= True)

    save_file=os.path.join(ROOT, config.data['output_path'], config.experiment['name'], config.experiment['version'], 'test_conf_matrix.png')
    logger = Logger(save_dir=os.path.join(ROOT, config.data['output_path'], config.experiment['name'], config.experiment['version']))
    val_metrics = validate_model("test", test_loader, device, model, loss_func, config.model['class_labels'], logger, save_file)

    return val_metrics


if __name__ == "__main__":
    config_path = os.path.join(ROOT, "config/unet_config.yml")
    config = load_config(config_path)
    model_path = "/home/husammm/Desktop/courses/cs_courses/DL/projects/semantic_segmentation/modeling/outputs/epoch2_model.pth"
    
    test_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    loss_func = nn.CrossEntropyLoss(ignore_index=255)
    
    test_metrics = test_model(config, model_path, loss_func, test_transform)
    print(f"Test Loss {test_metrics['avg_loss']:.4f} | IOU per class {test_metrics['iou_per_class']:.2f}% | Accuarcy per class {test_metrics['acc_per_class']:.4}")
