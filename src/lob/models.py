"""DeepLOB (Zhang, Zohren and Roberts, 2019) and a small transformer, in PyTorch.

Both take a [batch, 1, 100, 40] window of order book events and return logits for
up / stationary / down.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _conv(cin, cout, k, s=(1, 1), pad=0):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride=s, padding=pad),
                         nn.LeakyReLU(0.01), nn.BatchNorm2d(cout))


class DeepLOB(nn.Module):
    def __init__(self, n_classes: int = 3, lstm_hidden: int = 64):
        super().__init__()
        # 1) combine price and volume at each level, 2) bid with ask, 3) all 10 levels
        self.block1 = nn.Sequential(_conv(1, 32, (1, 2), (1, 2)),
                                    _conv(32, 32, (4, 1), pad="same"), _conv(32, 32, (4, 1), pad="same"))
        self.block2 = nn.Sequential(_conv(32, 32, (1, 2), (1, 2)),
                                    _conv(32, 32, (4, 1), pad="same"), _conv(32, 32, (4, 1), pad="same"))
        self.block3 = nn.Sequential(_conv(32, 32, (1, 10)),
                                    _conv(32, 32, (4, 1), pad="same"), _conv(32, 32, (4, 1), pad="same"))
        # inception module over time
        self.inc1 = nn.Sequential(_conv(32, 64, (1, 1)), _conv(64, 64, (3, 1), pad="same"))
        self.inc2 = nn.Sequential(_conv(32, 64, (1, 1)), _conv(64, 64, (5, 1), pad="same"))
        self.inc3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)), _conv(32, 64, (1, 1)))
        self.lstm = nn.LSTM(192, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))           # [B, 32, T, 1]
        x = torch.cat([self.inc1(x), self.inc2(x), self.inc3(x)], dim=1)  # [B, 192, T, 1]
        x = x.squeeze(-1).transpose(1, 2)                       # [B, T, 192]
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class LOBTransformer(nn.Module):
    def __init__(self, n_features: int = 40, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, window: int = 100, n_classes: int = 3):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        pos = torch.arange(window).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(window, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x):
        x = self.embed(x.squeeze(1)) + self.pe
        x = self.encoder(x)
        return self.head(x[:, -1])


def build(name: str) -> nn.Module:
    return {"deeplob": DeepLOB, "transformer": LOBTransformer}[name]()


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
