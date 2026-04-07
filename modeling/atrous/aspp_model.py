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
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1), 
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1)
        )
        
        self.enc_proj = nn.Sequential(
            nn.Conv2d(in_channels=(len(rates)+1)*512, out_channels=1024, kernel_size=1,),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, 
                               kernel_size=(2, 2), stride=8),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=4),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=256+512, out_channels=256,
                      kernel_size=1), 
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=1), 
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            ## Upsample
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=(2, 2), stride=4),
            nn.Conv2d(in_channels=64, out_channels=out_classes, kernel_size=(3, 3), padding=1),
        )
        
        # for child in self.entry:
        #     print(child)
