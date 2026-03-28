import torch
import torch.nn as nn
from model.blocks.ViT import ViTBlock

class CNNViT(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.vit = ViTBlock(dim=c_out, depth=8, heads=4, mlp_dim=32, patch_size=4)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_cnn = self.conv(x)  # [B,C,H,W]
        x_vit = self.vit(x_cnn, is_transpose=True)
        return x_cnn + x_vit * self.gamma