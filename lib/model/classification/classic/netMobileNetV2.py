import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConvConfig:
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.kernel_size = kernel_size

class MBConv(nn.Module):
    def __init__(self, config):
        super(MBConv, self).__init__()
        hidden_dim = config.in_channels * config.expand_ratio
        self.expand_conv = nn.Conv2d(config.in_channels, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=config.kernel_size, stride=config.stride,
                                        padding=config.kernel_size // 2, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.project_conv = nn.Conv2d(hidden_dim, config.out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(config.out_channels)

    def forward(self, x):
        residual = x
        x = F.relu6(self.bn1(self.expand_conv(x)))
        x = F.relu6(self.bn2(self.depthwise_conv(x)))
        x = self.bn3(self.project_conv(x))

        if residual.shape == x.shape:
            x = x + residual
        return x
