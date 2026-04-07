import torch
from torch import nn

from torchvision import models


class ASPP(nn.Module):
    def __init__(self, out_classes: int, rates:list=[1, 6, 12, 18]):
        super().__init__()

        self.resnet = models.resnet50(weights="DEFAULT")
        self.entry = nn.Sequential(*list(self.resnet.children())[:5])
        self.resnet = nn.Sequential(*list(self.resnet.children())[5:8])
        
        # for child in self.entry:
        #     print(child)
