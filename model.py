import torch
import torch.nn as nn


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
    
class UNet(nn.Module):
    def __init__(self, in_channels=8, num_classes=2):
        super().__init__()

        # --Encoding Phase -- 

        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # --Bottleneck (deepset poitn - no pooling here)

        self.bottleneck = ConvBlock(512, 1024)

        # --Decoding Phase --

        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        # -- Final Outout Layer -- 
        self.output_conv = nn.conv2d(64, num_classes, kernel_size=1)

        def forward(self, x):
            # ---Encoder---
            skip1, x = self.enc1(x)
            skip2, x = self.enc2(x)
            skip3, x = self.enc2(x)
            skip4, x = self.enc2(x)

            # ---Bottleneck---
            x = self.bottleneck(x)

            #---Decoder---
            x = self.dec4(x, skip4)
            x = self.dec3(x, skip3)
            x = self.dec2(x, skip2)
            x = self.dec1(x, skip1)

            # Final output

            return self.output_conv(x)