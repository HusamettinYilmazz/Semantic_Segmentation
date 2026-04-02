import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from modeling.unet import UNet
from .train import validate_model

from utils.dataset import VOCDataset
from utils import load_config



ROOT = ""


def test_model(config, model_path, loss_func, test_transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check_point = torch.load(model_path, map_location= device)

    dataset_path = os.path.join(ROOT, config.data['videos_path'])
    annot_path = os.path.join(ROOT, config.data['annot_path'])

    model = UNet(out_features=config.model['num_classes'])
    model.load_state_dict(check_point['model_state_dict'])
    model.to(device)

    test_dataset = VOCDataset(dataset_path, annot_path, split=config.data['video_splits']['test'], seq=config.experiment['sequential'], crop=config.experiment['crop'], all_players_once=config.experiment['all_players_once'], transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.training['batch_size'], shuffle= True, pin_memory=True)

    save_file=os.path.join(ROOT, config.data['output_path'], config.experiment['name'], config.experiment['version'], 'test_conf_matrix.png')
    val_metrics = validate_model(test_loader, device, model, loss_func, config.model['class_labels'], save_file)

    return val_metrics


if __name__ == "__main__":
    config_path = os.path.join(ROOT, "configs/unet_config.yaml")
    config = load_config(config_path)
    model_path = ""
    
    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    loss_func = nn.CrossEntropyLoss()
    
    test_metrics = test_model(config, model_path, loss_func, test_transform)
    print(f" Test Loss {test_metrics["avg_loss"]:.4f} | IOU per class {test_metrics["iou_per_class"]:.2f}% | Accuarcy per class {test_metrics["acc_per_class"]:.4}")
