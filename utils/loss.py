import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import PIXEL_VALUE_MAX
from utils.metrics import ssim
from torchvision.models import vgg19

class GaussianNLLLoss2D(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_mean, pred_var, target):
        # Ensure (B, 1, H, W)
        if pred_mean.ndim == 3:
            pred_mean = pred_mean.unsqueeze(1)
        if pred_var.ndim == 3:
            pred_var = pred_var.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        # 🔄 Resize target if needed
        if target.shape[-2:] != pred_mean.shape[-2:]:
            target = F.interpolate(target, size=pred_mean.shape[-2:], mode='bilinear', align_corners=False)

        # ✅ Clamp variance to avoid log(0)
        var = pred_var.clamp(min=self.eps)

        # 📉 Compute NLL
        nll = 0.5 * (torch.log(var) + ((target - pred_mean) ** 2) / var)

        # 🔁 Reduction
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll  # shape (B, 1, H, W)


class SSIMLoss(nn.Module):
    def __init__(self, max_val=PIXEL_VALUE_MAX, window_size=11):
        super(SSIMLoss, self).__init__()
        self.max_val = max_val
        self.window_size = window_size

    def forward(self, img1, img2):
        ssim_val = ssim(img1, img2, max_val=self.max_val, window_size=self.window_size)
        return 1 - ssim_val

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.epsilon**2))
        return loss

class GradientDifferenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sr, hr):
        grad_x_sr = sr[:, :, :, 1:] - sr[:, :, :, :-1]
        grad_x_hr = hr[:, :, :, 1:] - hr[:, :, :, :-1]
        grad_y_sr = sr[:, :, 1:, :] - sr[:, :, :-1, :]
        grad_y_hr = hr[:, :, 1:, :] - hr[:, :, :-1, :]
        return torch.mean(torch.abs(grad_x_sr - grad_x_hr)) + \
            torch.mean(torch.abs(grad_y_sr - grad_y_hr))

class PerceptualLoss(nn.Module):
    def __init__(self, layer_name='relu4_3'):
        super().__init__()
        vgg = vgg19(weights='IMAGENET1K_V1').features.eval()
        self.selected_layer = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 17,
            'relu4_3': 26
        }[layer_name]
        self.vgg = nn.Sequential(*list(vgg[:self.selected_layer+1]))
        for p in self.vgg.parameters():
            p.requires_grad = False  # freeze VGG

    def forward(self, sr, hr):
        # chuẩn hoá input theo ImageNet
        def norm(x):
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
            return (x - mean) / std

        # nếu ảnh là grayscale => chuyển sang RGB (3 kênh)
        if sr.shape[1] == 1:
            sr = sr.repeat(1, 3, 1, 1)
            hr = hr.repeat(1, 3, 1, 1)

        sr_vgg = self.vgg(norm(sr))
        hr_vgg = self.vgg(norm(hr))
        return F.l1_loss(sr_vgg, hr_vgg)
