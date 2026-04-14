import torch
import torch.nn as nn
import torch.nn.functional as F

class SETR_MLA(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 num_classes=21,
                 embed_dim=768):

        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.pos_encoding = PositionalEncoding(max_len= (img_size // patch_size)**2, d_model=embed_dim)
        self.encoder = ViTEncoder(depth=24, dim=embed_dim)
        self.decoder = MLAHead(embed_dim, 256, num_classes)

    def forward(self, x):
        x, hw = self.patch_embed(x)   # (B, HW, C)
        x = self.pos_encoding(x)
        features = self.encoder(x)

        out = self.decoder(features, hw)

        # upsample to original image size
        out = F.interpolate(out,
                            size=(self.img_size, self.img_size),
                            mode='bilinear',
                            align_corners=False)

        return out


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

class TransformerLayer(nn.Module):
    def __init__(self, dim, heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x),
                          self.norm1(x),
                          self.norm1(x))[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, depth=24, dim=768):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerLayer(dim) for _ in range(depth)
        ])

    def forward(self, x):
        features = []

        for i, blk in enumerate(self.blocks):
            x = blk(x)

            # pick layers for MLA (example: 6, 12, 18, 24)
            if i in [5, 11, 17, 23]:
                features.append(x)

        return features

class MLAHead(nn.Module):
    def __init__(self, embed_dim=768, out_channels=256, num_classes=21):
        super().__init__()

        # project each level
        self.proj = nn.ModuleList([
            nn.Conv2d(embed_dim, out_channels, 1)
            for _ in range(4)
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, features, hw):
        # features: list of (B, HW, C)
        H, W = hw
        outs = []

        for i, feat in enumerate(features):
            B, N, C = feat.shape

            # reshape to 2D
            feat = feat.transpose(1, 2).reshape(B, C, H, W)

            feat = self.proj[i](feat)

            # upsample to largest scale
            feat = F.interpolate(feat,
                                 size=(H, W),
                                 mode='bilinear',
                                 align_corners=False)

            outs.append(feat)

        x = torch.cat(outs, dim=1)
        x = self.fuse(x)
        x = self.classifier(x)

        return x
