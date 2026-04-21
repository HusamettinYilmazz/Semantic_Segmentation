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

class SpatialReductionAttention(nn.Module):
    def __init__(self, emb_dim, heads=8, sr_ratio=1):
        super().__init__()
        self.sr_ratio = sr_ratio
        self.att = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=heads,
            batch_first=True,
        )
        if sr_ratio > 1:
            self.sr = nn.Conv2d(emb_dim, emb_dim,
                                kernel_size=sr_ratio,
                                stride=sr_ratio)
            self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = x
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x

        k, v = x_, x_
        attn_out, _ = self.att(q, k, v)

        return attn_out
