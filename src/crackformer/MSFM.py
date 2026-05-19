import torch
from torch import nn
import torch.nn.functional as F


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(CBR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class ASDW(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, dilate, bias=False):
        super(ASDW, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilate,
                      bias=bias, groups=in_channels)
        )

    def forward(self, x):
        return self.block(x)


class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super(SAFM, self).__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        ds = [4, 3, 2, 1]
        self.mfr = nn.ModuleList([
            nn.Conv2d(chunk_dim, chunk_dim, 3, 1, padding=ds[i], groups=chunk_dim, dilation=ds[i])
            for i in range(n_levels)
        ])
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x):

        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='bilinear', align_corners=True)
            else:
                s = self.mfr[i](xc[i])
            out.append(s)
        out = self.aggr(torch.cat(out, dim=1)) + x
        out = self.act(out)
        return out


class FusionBlock2(nn.Module):
    def __init__(self, f_in, x_inc):
        super(FusionBlock2, self).__init__()
        self.block = CBR(f_in + x_inc, x_inc, 3, 1, 1, bias=True)
        self.enhance = SAFM(x_inc)
        self.drop = nn.Dropout(0.1)

    def forward(self, fin, xin):
        x = torch.cat([fin, xin], 1)
        x = self.block(x)
        x = self.enhance(x)
        x = self.drop(x) + xin
        return x


class FusionBlock3(nn.Module):
    def __init__(self, f_in, f2_in, x_inc):
        super(FusionBlock3, self).__init__()
        self.block = CBR(f_in + f2_in + x_inc, x_inc, 3, 1, 1, bias=True)
        self.enhance = SAFM(x_inc)
        self.drop = nn.Dropout(0.1)
    def forward(self, fin, xin, fin2):
        x = torch.cat([fin, xin, fin2], 1)
        x = self.block(x)
        x = self.enhance(x)
        x = self.drop(x) + xin
        return x
        

class FusionModule(nn.Module):
    def __init__(self, channels: list = [64, 128, 256, 512]):
        super(FusionModule, self).__init__()
        self.fusion = nn.ModuleList()
        self.channels = channels
        for i, ch in enumerate(channels):
            if i == 0:
                self.fusion.append(
                    FusionBlock2(f_in=channels[i + 1], x_inc=channels[i])
                )
            elif i == len(channels) - 1:
                self.fusion.append(
                    FusionBlock2(f_in=channels[i - 1], x_inc=channels[i])
                )
            else:
                self.fusion.append(
                    FusionBlock3(f_in=channels[i - 1],
                                 f2_in=channels[i + 1],
                                 x_inc=channels[i]))

    def forward(self, feats):
        for i in range(len(feats)):
            if i == 0:
                fu = F.interpolate(feats[1], feats[i].shape[2:], mode='bilinear', align_corners=True)
                feats[i] = self.fusion[i](fu, feats[i])
            elif i == len(feats) - 1:
                fd = F.interpolate(feats[-2], feats[i].shape[2:], mode='bilinear', align_corners=True)
                feats[i] = self.fusion[i](fd, feats[i])
            else:
                fd = F.interpolate(feats[i - 1], feats[i].shape[2:], mode='bilinear', align_corners=True)
                fu = F.interpolate(feats[i + 1], feats[i].shape[2:], mode='bilinear', align_corners=True)
                feats[i] = self.fusion[i](fd, feats[i], fu)
        return feats