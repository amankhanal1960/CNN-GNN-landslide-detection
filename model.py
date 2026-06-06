import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
            nn.ReLU(implace=True)
        )

    def forward(self, x):
        features = self.block(x)
        return features


# ---- Encoder Block ---- #

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled

# ---- Decoder Block ---- #

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x , skip):
        upsampled = self.upsample(x)
        cat = torch.cat([upsampled, skip], dim=1)
        x = self.conv(cat)
        return x

