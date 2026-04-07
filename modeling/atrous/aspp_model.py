import torch
from torch import nn

from torchvision import models


class ASPP(nn.Module):
    def __init__(self, out_classes: int, rates:list=[1, 6, 12, 18]):
        super().__init__()

        self.resnet = models.resnet50(weights="DEFAULT")
        self.entry = nn.Sequential(*list(self.resnet.children())[:5])
        self.resnet = nn.Sequential(*list(self.resnet.children())[5:8])
        self.scales = nn.ModuleList()
        for rate in rates:
            network = nn.Sequential(
                nn.Conv2d(in_channels= 2048, out_channels= 512,
                          kernel_size= 3 if rate != 1 else 1, 
                          padding= rate if rate != 1 else 0,
                          dilation=rate),
                          nn.BatchNorm2d(num_features=512), ## out_channels
                          nn.ReLU()
            )
            self.scales.append(network)
       
        # for child in self.entry:
        #     print(child)
