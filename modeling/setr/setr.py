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

