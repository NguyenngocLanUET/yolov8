import torch
import torch.nn as nn

def autopad(k, p=None):
    return p if p is not None else k // 2


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3)
        self.cv2 = Conv(c_, c2, 3)
        self.add = c1 == c2

    def forward(self, x):
        if self.add:
            return x + self.cv2(self.cv1(x))
        return self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, self.c * 2, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))