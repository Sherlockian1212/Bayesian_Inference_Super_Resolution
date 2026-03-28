import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks.CNNViT import CNNViT
from model.blocks.residual import ResidualBlock
from model.blocks.upscale import UpscaleModule
from model.blocks.bayesian_head import BayesianHead
from utils import config
from torchinfo import summary

class ModelSR(nn.Module):
    def __init__(self, c=config.CHANNELS, mid_channel=16, h=128, w=128, scale=config.DOWN_SCALE_RATE):
        super().__init__()
        self.scale = scale
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c*mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c*mid_channel),
            nn.LeakyReLU()
        )

        self.residual_1 = nn.Sequential(*[ResidualBlock(c*mid_channel) for _ in range(16)])
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=c*mid_channel, out_channels=c*mid_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c*mid_channel),
            nn.LeakyReLU()
        )

        self.residual_cnn_vit = nn.Sequential(*[CNNViT(c_in=c*mid_channel, c_out=c*mid_channel) for _ in range(16)])

        self.bayesianHead =  BayesianHead()

        self.up_x2 = UpscaleModule(in_ch_up=mid_channel, in_ch_fused=mid_channel)
        self.residuals_x2 = ResidualBlock(c)

        if self.scale >= 4:
            self.up_x4 = UpscaleModule(in_ch_up=mid_channel, in_ch_fused=mid_channel)
            self.residuals_x4 = ResidualBlock(c)

        if self.scale >= 8:
            self.up_x8 = UpscaleModule(in_ch_up=mid_channel, in_ch_fused=mid_channel)
            self.residuals_x8 = ResidualBlock(c)

    def forward(self, x, sample=True):
        mid_1 = self.conv_1(x)
        feature_extraction_1 = self.residual_1(mid_1)

        mid_2 = self.conv_2(feature_extraction_1)
        feature_extraction_2 = self.residual_cnn_vit(mid_2)

        # x2 upscale
        out_x2_1, out_x2_2 = self.up_x2(feature_extraction_1, feature_extraction_2)

        # x4 upscale
        if self.scale == 4:
            feature_extraction_1 = F.interpolate(feature_extraction_1, scale_factor=2, mode='bilinear', align_corners=False)
            feature_extraction_2 = F.interpolate(feature_extraction_2, scale_factor=2, mode='bilinear', align_corners=False)
            out_x4_1, out_x4_2 = self.up_x4(feature_extraction_1, feature_extraction_2)
            out_x4, sigma = self.bayesianHead(out_x4_1, out_x4_2, sample=sample)
            out = self.residuals_x4(out_x4)
            return torch.clamp(out, 0.0, 1.0), sigma

        # x8 upscale
        elif self.scale == 8:
            feature_extraction_1 = F.interpolate(feature_extraction_1, scale_factor=4, mode='bilinear', align_corners=False)
            feature_extraction_2 = F.interpolate(feature_extraction_2, scale_factor=4, mode='bilinear', align_corners=False)
            out_x8_1, out_x8_2 = self.up_x8(feature_extraction_1, feature_extraction_2)
            out_x8, sigma = self.bayesianHead(out_x8_1, out_x8_2, sample=sample)
            out = self.residuals_x8(out_x8)
            return torch.clamp(out, 0.0, 1.0), sigma

        # x2 upscale (default)
        else:
            out_x2, sigma = self.bayesianHead(out_x2_1, out_x2_2, sample=sample)
            out = self.residuals_x2(out_x2)
            return torch.clamp(out, 0.0, 1.0), sigma


if __name__ == "__main__":
    # ==== Test Generator ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = ModelSR(c=1, h=128, w=128).to(device)

    summary(gen, input_size=(1, 1, 128, 128), device=device.type, sample=False)

    torch.cuda.reset_peak_memory_stats()
    input_size = (32, 1, 128, 128)
    x = torch.randn(*input_size).to(device)

    with torch.no_grad():
        gen_out, _ = gen(x)

    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"🚀 Peak GPU Memory Usage (Generator): {peak:.2f} MB")

    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(f"🧠 GPU Memory Allocated: {allocated:.2f} MB")
        print(f"📦 GPU Memory Reserved : {reserved:.2f} MB")
    else:
        print("💻 Using CPU, no GPU memory usage to display.")