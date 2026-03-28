import torch
import torch.nn.functional as F
import math
from torchvision.transforms.functional import gaussian_blur
from utils.config import PIXEL_VALUE_MAX


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = PIXEL_VALUE_MAX) -> float:
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img1, img2: Tensors of shape [C, H, W] or [B, C, H, W], with values in [0, max_val]
        max_val: Maximum possible pixel value (usually 1.0 or 255)

    Returns:
        PSNR value in dB
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr_val = 20 * math.log10(max_val) - 10 * math.log10(mse.item())
    return psnr_val

def ssim(img1: torch.Tensor, img2: torch.Tensor, max_val: float = PIXEL_VALUE_MAX, window_size: int = 11):
    """
    Compute SSIM (Structural Similarity Index) between two images.

    Args:
        img1, img2: Tensors of shape [C, H, W] or [B, C, H, W], values in [0, max_val]
        max_val: Maximum pixel value (default: PIXEL_VALUE_MAX)
        window_size: Size of Gaussian filter (default: 11)

    Returns:
        SSIM value in [0, 1]
    """
    def _gaussian_filter(x):
        return gaussian_blur(x, kernel_size=window_size, sigma=1.5)

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu1 = _gaussian_filter(img1)
    mu2 = _gaussian_filter(img2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _gaussian_filter(img1 * img1) - mu1_sq
    sigma2_sq = _gaussian_filter(img2 * img2) - mu2_sq
    sigma12 = _gaussian_filter(img1 * img2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()
