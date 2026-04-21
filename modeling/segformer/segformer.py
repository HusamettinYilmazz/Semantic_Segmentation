import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, emb_dim, patch_size=7, overlap_size=3):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_ch,
            out_channels= emb_dim,
            kernel_size=patch_size,
            stride=patch_size - overlap_size,
            padding=overlap_size
            )   ## Original paper values: patch_size=7, stride=4, padding=3
        
        self.norm = nn.LayerNorm(emb_dim)
    def forward(self, x):
        x = self.proj(x)    ## [B, C, H, W] it isn't H and W, it is N= num_of_patches
        x = x.flatten(2).transpose(1, 2)    ## [B, N, C]
        out = self.norm(x)
        return out
