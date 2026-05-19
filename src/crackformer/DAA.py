import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class DualAxialAttention_Block(nn.Module):
    def __init__(self, inp, reduction=32):
        super(DualAxialAttention_Block, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = inp // reduction

        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(inp),
            h_swish()
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0),
            h_swish(),
            nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0),
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0),
            h_swish(),
            nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = x * a_w * a_h
        return out

if __name__ == '__main__':
    model = DualAxialAttention_Block(64)
    x = torch.randn(1, 64, 224, 224)
    output = model(x)
    print(output.size())
