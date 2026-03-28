import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, expansion=4):
        super().__init__()
        mid_channels = channels * expansion
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.prelu1(self.conv1(x))
        out = self.prelu2(self.conv2(out))
        out = self.conv3(out)
        return residual + out