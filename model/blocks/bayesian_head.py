import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks.bayes_conv2d import BayesConv2d

class BayesianHead(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.conv_mu   = BayesConv2d(c * 2, 1, kernel_size=1)
        self.conv_rho  = BayesConv2d(c * 2, 1, kernel_size=1)

    def forward(self, x1, x2, sample=True):
        fused = torch.cat([x1, x2], dim=1)
        mu   = self.conv_mu(fused, sample)
        rho  = self.conv_rho(fused, sample)
        sigma = F.softplus(rho) + 1e-6
        out = torch.sigmoid(mu)
        sigma = sigma
        return out, sigma

    def kl_loss(self):
        return self.conv_mu.kl_loss() + self.conv_rho.kl_loss()