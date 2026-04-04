import torch
import torch.nn as nn
from torchvision import models

import torchvision


class ResUNet(nn.Module):
    def __init__(self, out_classes):
        super().__init__()
        self.resnet50 = models.resnet50(weights="DEFAULT")
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])

        self.enc_block0 = nn.Sequential(*list(self.resnet50.children())[:4])
        self.enc_block1 = nn.Sequential(*list(self.resnet50.children())[4:5])
        self.enc_block2 = nn.Sequential(*list(self.resnet50.children())[5:6])
        self.enc_block3 = nn.Sequential(*list(self.resnet50.children())[6:7])
        self.enc_block4 = nn.Sequential(*list(self.resnet50.children())[7:8])
        

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=self.enc_block4[-1][-1]._modules["conv3"].out_channels, 
                      out_channels=2 * self.enc_block4[-1][-1]._modules["conv3"].out_channels, 
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2 * self.enc_block4[-1][-1]._modules["conv3"].out_channels, 
                      out_channels=2 * self.enc_block4[-1][-1]._modules["conv3"].out_channels, 
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=2 * self.enc_block4[-1][-1]._modules["conv3"].out_channels, 
                               out_channels=self.enc_block4[-1][-1]._modules["conv3"].out_channels, 
                               kernel_size=(2, 2), stride=2)
        )
        

        self.dec_block4 = nn.Sequential(
            nn.Conv2d(in_channels= self.enc_block4[-1][-1]._modules["conv3"].out_channels + self.bottleneck[-1].out_channels, 
                      out_channels=self.bottleneck[-1].out_channels, 
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.bottleneck[-1].out_channels, 
                      out_channels=self.bottleneck[-1].out_channels, 
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.bottleneck[-1].out_channels, 
                               out_channels=self.bottleneck[-1].out_channels//2, 
                               kernel_size=(2, 2), stride=2)
        )
        self.dec_block3 = nn.Sequential(
            nn.Conv2d(in_channels= self.enc_block3[-1][-1]._modules["conv3"].out_channels + self.dec_block4[-1].out_channels, 
                      out_channels=self.dec_block4[-1].out_channels, 
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dec_block4[-1].out_channels, 
                      out_channels=self.dec_block4[-1].out_channels, 
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.dec_block4[-1].out_channels, 
                               out_channels=self.dec_block4[-1].out_channels//2, 
                               kernel_size=(2, 2), stride=2),
        )
        self.dec_block2 = nn.Sequential(
            nn.Conv2d(in_channels= self.enc_block2[-1][-1]._modules["conv3"].out_channels + self.dec_block3[-1].out_channels, 
                      out_channels=self.dec_block3[-1].out_channels, 
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dec_block3[-1].out_channels, 
                      out_channels=self.dec_block3[-1].out_channels, 
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.dec_block3[-1].out_channels, 
                               out_channels=self.dec_block3[-1].out_channels//2, 
                               kernel_size=(2, 2), stride=2),
        )
        self.dec_block1 = nn.Sequential(
            nn.Conv2d(in_channels= self.enc_block1[-1][-1]._modules["conv3"].out_channels + self.dec_block2[-1].out_channels, 
                      out_channels=self.dec_block2[-1].out_channels, 
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dec_block2[-1].out_channels, 
                      out_channels=self.dec_block2[-1].out_channels, 
                      kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.dec_block2[-1].out_channels, 
                               out_channels=self.dec_block2[-1].out_channels//2, 
                               kernel_size=(2, 2), stride=2),
        
        )
        self.dec_block0 = nn.Sequential(
            nn.Conv2d(in_channels=self.enc_block0[0].out_channels + self.dec_block1[-1].out_channels, 
                    out_channels=self.dec_block1[-1].out_channels, 
                    kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dec_block1[-1].out_channels, 
                    out_channels=self.dec_block1[-1].out_channels, 
                    kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.dec_block1[-1].out_channels,
                            out_channels=self.dec_block1[-1].out_channels//2,
                            kernel_size=(2, 2), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dec_block1[-1].out_channels//2, 
                    out_channels=self.dec_block1[-1].out_channels//2, 
                    kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dec_block1[-1].out_channels//2, 
                    out_channels=out_classes, 
                    kernel_size=(1, 1)),
        )

        # for child in list(self.resnet50.children())[:]:
        #     print(child)

    def forward(self, x):
        # print(x.shape)

        enc0_x = self.enc_block0(x)
        enc1_x = self.enc_block1(enc0_x)
        enc2_x = self.enc_block2(enc1_x)
        enc3_x = self.enc_block3(enc2_x)    ## [1, 1024, 18, 32] I believe its ok but bigger is better
        enc4_x = self.enc_block4(enc3_x)    ## [1, 2048, 9, 16] width and height are too small
        
        bottleneck_x = self.bottleneck(enc4_x)
        
        dec4_x = self.dec_block4(torch.cat((bottleneck_x, torchvision.transforms.functional.center_crop(enc4_x, bottleneck_x.shape[2:])), dim=1))
        dec3_x = self.dec_block3(torch.cat((dec4_x, torchvision.transforms.functional.center_crop(enc3_x, dec4_x.shape[2:])), dim=1))
        dec2_x = self.dec_block2(torch.cat((dec3_x, torchvision.transforms.functional.center_crop(enc2_x, dec3_x.shape[2:])), dim=1))
        dec1_x = self.dec_block1(torch.cat((dec2_x, torchvision.transforms.functional.center_crop(enc1_x, dec2_x.shape[2:])), dim=1))
        dec0_x = self.dec_block0(torch.cat((dec1_x, torchvision.transforms.functional.center_crop(enc0_x, dec1_x.shape[2:])), dim=1))
        # print(dec1_x.shape)

        return dec0_x



"""
TODO later:
    - The image resoluation at botteleneck block is too small 
    I may remove self.enc_block4 and self.dec_block4 and fix self.botteleneck
    input and output to match self.enc_block3 and self.dec_block3
"""