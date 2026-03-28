import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks.se import SEBlock

class UpscaleModule(nn.Module):
    """
    Upscale factor fixed to 2 (pixelshuffle 2) — combined:
      - concat(up, fused) -> multi-branch convs (1x1,3x3,5x5)
      - fuse -> produce channels for PixelShuffle
      - SE attention
      - PixelShuffle branch + ConvTranspose branch (deconv)
      - residual skip (bilinear upsample of `up`)
      - final refine conv
    Args:
      in_ch_up: channels of `up` tensor (low-res feature)
      in_ch_fused: channels of `fused` tensor (context/high-res feature)
      out_ch: desired output channels after upscaling (e.g. 3 for RGB)
      mid_ch: internal channel width (bigger -> more params)
      reduction: SE reduction factor
    """
    def __init__(self, in_ch_up=1, in_ch_fused=2, out_ch=1, mid_ch=128, reduction=16):
        super().__init__()
        self.upscale = 2
        self.in_ch_up = in_ch_up
        self.in_ch_fused = in_ch_fused
        self.out_ch = out_ch
        self.mid_ch = mid_ch

        concat_ch = in_ch_up + in_ch_fused

        # initial projection of concat -> mid_ch
        self.proj = nn.Sequential(
            nn.Conv2d(concat_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # SE attention applied on pre-ps features (mid_ch)
        self.se = SEBlock(mid_ch, reduction=reduction)

        # fuse to channels needed for pixelshuffle (out_ch * 4 for x2)
        self.fuse_to_ps = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch * (self.upscale**2), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch * (self.upscale**2)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # PixelShuffle branch
        self.pixel_shuffle = nn.PixelShuffle(self.upscale)

        # Deconv branch (ConvTranspose2d) - provides complementary learned upsampling
        # For x2 common kernel=4, stride=2, padding=1
        self.deconv = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        )

        # refine conv
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        )

        # final small residual scaling
        self.res_scale = 1.0

        # init weights
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, up, fused):
        """
        up: [B, in_ch_up, H, W]  (residual feature)
        fused: [B, in_ch_fused, H/2, W/2] (CNNViT feature)
        We'll interpolate fused to up's spatial size before concat.
        Returns: [B, out_ch, H*2, W*2]
        """

        # align spatial
        if fused.shape[2:] != up.shape[2:]:
            fused = F.interpolate(fused, size=up.shape[2:], mode='bilinear', align_corners=False)

        # concat
        x = torch.cat([up, fused], dim=1)   # [B, concat_ch, H, W]
        x = self.proj(x)                    # [B, mid_ch, H, W]

        # SE attention
        x_att = self.se(x)               # [B, mid_ch, H, W]

        # pixelshuffle branch
        ps_pre = self.fuse_to_ps(x_att)    # [B, out_ch*4, H, W]
        ps = self.pixel_shuffle(ps_pre)    # [B, out_ch, H*2, W*2]

        # deconv branch: operate on x_m -> upsample to out_ch
        deconv_out = self.deconv(x_att)    # [B, out_ch, H*2, W*2]

        # residual skip (interpolated up)
        skip = F.interpolate(up, scale_factor=2, mode='bilinear', align_corners=False)  # [B, in_ch_up, H*2, W*2]
        # project skip to out_ch if channels differ
        if skip.shape[1] != self.out_ch:
            skip = F.interpolate(skip, size=skip.shape[2:], mode='nearest')  # keep shape
            # 1x1 conv to map channels (do on-the-fly)
            skip = torch.nn.functional.conv2d(skip, weight=torch.eye(self.out_ch, skip.shape[1]).view(self.out_ch, skip.shape[1], 1, 1).to(skip.device))

        ps = ps + skip * 0.3
        deconv_out = deconv_out + skip * 0.3

        return  ps, deconv_out