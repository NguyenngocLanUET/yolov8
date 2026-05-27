import torch
import torch.nn as nn
from .blocks import Conv, C2f


class PAN(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_p5_to_p4 = Conv(ch[2], ch[1], 1)
        self.c2f_p4 = C2f(ch[1] * 2, ch[1], n=1)

        self.conv_p4_to_p3 = Conv(ch[1], ch[0], 1)
        self.c2f_p3 = C2f(ch[0] * 2, ch[0], n=1)

        self.down_n3_to_n4 = Conv(ch[0], ch[0], 3, 2)
        self.c2f_n4 = C2f(ch[0] + ch[1], ch[1], n=1)

        self.down_n4_to_n5 = Conv(ch[1], ch[1], 3, 2)
        self.c2f_n5 = C2f(ch[1] + ch[2], ch[2], n=1)

    def forward(self, x):
        p3, p4, p5 = x

        p4_mid = self.c2f_p4(
            torch.cat([p4, self.up(self.conv_p5_to_p4(p5))], 1)
        )

        n3 = self.c2f_p3(
            torch.cat([p3, self.up(self.conv_p4_to_p3(p4_mid))], 1)
        )

        n4 = self.c2f_n4(
            torch.cat([p4_mid, self.down_n3_to_n4(n3)], 1)
        )

        n5 = self.c2f_n5(
            torch.cat([p5, self.down_n4_to_n5(n4)], 1)
        )

        return [n3, n4, n5]