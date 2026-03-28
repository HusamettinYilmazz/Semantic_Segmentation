import torch
import torch.nn as nn
from torchvision import models

import torchvision

class UNet(nn.Module):
    def __init__(self, out_classes):
        super().__init__()
        # self.resnet50 = models.resnet50(weights="DEFAULT")
        # self.resnet50 = nn.Sequential(*list(self.resnet50.children())[5:6])

        self.enc_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            )
        self.enc_block2 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )
        self.enc_block3 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )
        self.enc_block4 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )

"""
What I've did:
    I saw the archtecture of resnet50
    I reviewed Conv layer components (in_channels= channels not width or height of the image)
    Conv layers are diemention invariant

    
    1. Build the sequence of the encoder of UNet:
        - (Later) There is something like block seperation in Resnet (based on dict most probably)
        see how to build the same architecture to the ecoder blocks(2 conv layer + max pooling)

"""