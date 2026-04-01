import os 
import yaml

import matplotlib.pyplot as plt
import torch
class Config:
    def __init__(self, config_dict):
        self.experiment = config_dict.get("experiment", {})
        self.data = config_dict.get("data", {})
        self.model = config_dict.get("model", {})
        self.training = config_dict.get("training", {})

    def __reper__(self):
        return f"Config(experiment={self.experiment}, data={self.data} model={self.model}, training={self.training})"
     

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = Config(config)
    return config

def save_checkpoint(epoch, model, optimizer, cur_lr, val_acc, config, train_transform, val_transform, save_dir):
    checkpoint_path = os.path.join(save_dir, f'epoch{epoch}_model.pth')

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': cur_lr,
            'val_accuracy': val_acc,
            'config': config,
            'train_transform': train_transform,
            'val_transform': val_transform

        }, checkpoint_path)

    print(f"Epoch:{epoch} checkpoint has been saved at:{checkpoint_path}")


def lr_vs_epoch(num_epochs, lrs, save_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), lrs, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs. Epoch")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'epoch_vs_lr.png'), bbox_inches='tight')