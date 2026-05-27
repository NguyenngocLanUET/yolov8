import torch
import torch.nn as nn
from .blocks import Conv


class DetectHead(nn.Module):
    def __init__(self, nc=5, ch=(256, 512, 1024), reg_max=16):
        super().__init__()

        self.nc = nc
        self.reg_max = reg_max
        self.nl = len(ch)

        self.cls_convs = nn.ModuleList([
            nn.Sequential(
                Conv(x, x, 3),
                nn.Conv2d(x, nc, 1)
            )
            for x in ch
        ])

        self.reg_convs = nn.ModuleList([
            nn.Sequential(
                Conv(x, x, 3),
                nn.Conv2d(x, 4 * reg_max, 1)
            )
            for x in ch
        ])

    def forward(self, x):
        cls_preds, reg_preds = [], []

        for i in range(self.nl):
            cls_preds.append(
                self.cls_convs[i](x[i]).flatten(2).transpose(1, 2)
            )
            reg_preds.append(
                self.reg_convs[i](x[i]).flatten(2).transpose(1, 2)
            )

        return torch.cat(cls_preds, 1), torch.cat(reg_preds, 1)