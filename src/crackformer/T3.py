import torch
from torch import nn
import torch.nn.functional as F

from src.crackformer.T3_2 import WeightedFeatureFusion


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x

class BalancedAttentionFuse(nn.Module):

    def __init__(self, embedding_dim):
        super(BalancedAttentionFuse, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim * 4, 3, padding=1, groups=embedding_dim * 4, bias=False),
            nn.BatchNorm2d(embedding_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),

        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embedding_dim, embedding_dim // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 8, embedding_dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        fused = self.fuse_conv(x)
        att = self.channel_attention(fused)
        output = fused * att
        return output

class EnhancedSegFormerHead(nn.Module):
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(EnhancedSegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.wff_c43 = WeightedFeatureFusion()
        self.wff_c432 = WeightedFeatureFusion()
        self.wff_c4321 = WeightedFeatureFusion()
        self.attention_fuse = BalancedAttentionFuse(embedding_dim)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3]).contiguous()
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3]).contiguous()
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3]).contiguous()
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3]).contiguous()

        _c4 = self.wff_c43(_c4, _c3)
        _c3 = self.wff_c432(_c3, _c2)
        _c2 = self.wff_c4321(_c2, _c1)

        _c = self.attention_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x