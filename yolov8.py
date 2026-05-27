import torch.nn as nn
from .backbone import Backbone
from .neck import PAN
from .head import DetectHead


class YOLOv8(nn.Module):
    def __init__(self, num_classes=5, reg_max=16):
        super().__init__()

        self.backbone = Backbone()
        self.neck = PAN(self.backbone.out_channels)

        self.head = DetectHead(
            nc=num_classes,
            ch=self.backbone.out_channels,
            reg_max=reg_max
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        preds = self.head(feats)
        return preds, feats