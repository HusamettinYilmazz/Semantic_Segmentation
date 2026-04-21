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

class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim  # depthwise
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        x = self.fc1(x)  # (B, N, hidden_dim)

        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dwconv(x)

        x = x.reshape(B, -1, H * W).transpose(1, 2)
        x = self.act(x)
        out = self.fc2(x)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.att = SpatialReductionAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mix_fnn = MixFFN(dim)

    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.att(x_norm)

        x_norm2 = self.norm2(x)
        out = x + self.mix_fnn(x_norm2)

        return out

class EncoderStage(nn.Module):
    def __init__(self, in_ch, embed_dim, depth, patch_size, stride):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_ch, embed_dim, patch_size, stride)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

class Encoder(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.stage1 = EncoderStage(3, 64, 2, 7, 4)
        self.stage2 = EncoderStage(64, 128, 2, 3, 2)
        self.stage3 = EncoderStage(128, 320, 2, 3, 2)
        self.stage4 = EncoderStage(320, 512, 2, 3, 2)

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(x)
        f3 = self.stage3(x)
        f4 = self.stage4(x)
        return f1, f2, f3, f4

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Linear(64, 256),
            nn.Linear(128, 256),
            nn.Linear(320, 256),
            nn.Linear(512, 256),
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(256 * 4, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, feats, out_size):
        outs = []
        for i, f in enumerate(feats):
            B, N, C = f.shape
            h = w = int(N ** 0.5)

            f = self.proj[i](f)
            f = f.transpose(1, 2).reshape(B, 256, h, w)

            f = F.interpolate(f, size=feats[0].shape[1] ** 0.5, mode="bilinear", align_corners=False)
            outs.append(f)

        x = torch.cat(outs, dim=1)
        x = self.fuse(x)

        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x
