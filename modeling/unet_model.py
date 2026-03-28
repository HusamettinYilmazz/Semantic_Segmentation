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
        
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2, 2), stride=2)
        )
        
        self.dec_block4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 2), stride=2)
        )
        self.dec_block3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2),
        )
        self.dec_block2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2),
        )
        self.dec_block1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=out_classes, kernel_size=(1, 1)),
        )

        # for child in list(self.resnet50.children())[6:7]:
        #     print(child)

    def forward(self, x):
        # print(x.shape)
        
        enc1_x = self.enc_block1(x)
        enc2_x = self.enc_block2(enc1_x)
        enc3_x = self.enc_block3(enc2_x)
        enc4_x = self.enc_block4(enc3_x)
        
        bottleneck_x = self.bottleneck(enc4_x)
        
        dec4_x = self.dec_block4(torch.cat((bottleneck_x, torchvision.transforms.functional.center_crop(enc4_x, bottleneck_x.shape[2:])), dim=1))
        dec3_x = self.dec_block3(torch.cat((dec4_x, torchvision.transforms.functional.center_crop(enc3_x, dec4_x.shape[2:])), dim=1))
        dec2_x = self.dec_block2(torch.cat((dec3_x, torchvision.transforms.functional.center_crop(enc2_x, dec3_x.shape[2:])), dim=1))
        dec1_x = self.dec_block1(torch.cat((dec2_x, torchvision.transforms.functional.center_crop(enc1_x, dec2_x.shape[2:])), dim=1))

        # print(dec1_x.shape)

        return dec1_x



"""
What I've did:
    I saw the archtecture of resnet50
    I reviewed Conv layer components (in_channels= channels not width or height of the image)
    Conv layers are diemention invariant

    
    1. Build the sequence of the encoder of UNet:
        - (Later) There is something like block seperation in Resnet (based on dict most probably)
        see how to build the same architecture to the ecoder blocks(2 conv layer + max pooling)

    2. Build the sequence of the decoder
        - Be sure if the deconve is the same to ConvTranspose2d or not (Watch dr mostafa lectures to remember theory)
        
        - The purpose is to (e.g. combine the c channels to c/2 channels(DONE) ---> (still) and go form h/2*w/2 to h*w)
    
    1. add pading = 1 to normal conv so the h_out = h_input - kernal + 1 + padding
        If I do so there will be no need to the crop I did (better idea instead of cropping).
        It still happens due to the input image size (Not divisble by 2)
        SOLUTION:   
            1. crop as I already did (Work on center cropping not top-left one)
            2. Force the input to by divisble by 2


What I should do next:
    1. Go to train.py and continue there.

"""