import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, prior_mu=0., prior_sigma=2.):
        super().__init__()
        self.stride = stride
        self.padding = padding

        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))

        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.register_buffer("prior_mu", torch.tensor(prior_mu))
        self.register_buffer("prior_sigma", torch.tensor(prior_sigma))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=5**0.5)
        nn.init.constant_(self.weight_rho, -3.0)
        if self.bias_mu is not None:
            nn.init.constant_(self.bias_mu, 0.0)
            nn.init.constant_(self.bias_rho, -3.0)

    def sample_weight(self):
        sigma_w = F.softplus(self.weight_rho) + 1e-6
        weight = self.weight_mu + sigma_w * torch.randn_like(sigma_w)
        if self.bias_mu is not None:
            sigma_b = F.softplus(self.bias_rho) + 1e-6
            bias = self.bias_mu + sigma_b * torch.randn_like(sigma_b)
        else:
            bias = None
        return weight, bias

    def forward(self, x, sample=True):
        weight, bias = self.sample_weight() if sample else (self.weight_mu, self.bias_mu)
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)


    def kl_loss(self):
        sigma_w = F.softplus(self.weight_rho) + 1e-6
        kl_w = torch.log(self.prior_sigma / sigma_w) + (sigma_w**2 + (self.weight_mu - self.prior_mu)**2) / (2 * self.prior_sigma**2) - 0.5
        kl = kl_w.sum()
        if self.bias_mu is not None:
            sigma_b = F.softplus(self.bias_rho) + 1e-6
            kl_b = torch.log(self.prior_sigma / sigma_b) + (sigma_b**2 + (self.bias_mu - self.prior_mu)**2) / (2 * self.prior_sigma**2) - 0.5
            kl += kl_b.sum()
        return kl