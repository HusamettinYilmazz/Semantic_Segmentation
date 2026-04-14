import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        positions = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * 
            (-torch.log(torch.tensor(1e4))/d_model)
        )
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] += torch.sin(positions * div)
        self.pe[:, 1::2] += torch.cos(positions * div)
        self.pe = self.pe.unsqueeze(0)

        self.register_buffer("pe", self.pe)


    def forward(self, x):
        ## x: (B, N, c)
        x += self.pe[:, :x.size(1)]
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        ## x: (B, C, H, W)
        x = self.proj(x)                     ## (B, E, H/P, W/P)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)     ## (B, HW, E)
        return x, (H, W)
