import torch
import torch.nn as nn
import time


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return skip, x


class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.conv = ConvBlock(out_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.up_conv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, features=(64, 128, 256, 512), bottleneck_channels=1024):
        super().__init__()

        self.down1 = Down(in_channels, features[0])
        self.down2 = Down(features[0], features[1])
        self.down3 = Down(features[1], features[2])
        self.down4 = Down(features[2], features[3])

        self.bottleneck = ConvBlock(features[3], bottleneck_channels)

    def forward(self, x):
        s1, x = self.down1(x)   # 64, 128x128 if input 256
        s2, x = self.down2(x)   # 128, 64x64
        s3, x = self.down3(x)   # 256, 32x32
        s4, x = self.down4(x)   # 512, 16x16
        b = self.bottleneck(x)  # 1024, 16x16 if input was 256 after 4 pools -> actually 16x16

        skips = [s1, s2, s3, s4]
        return b, skips


class Decoder(nn.Module):
    def __init__(self, out_classes=1, bottleneck_channels=1024, features=(64, 128, 256, 512)):
        super().__init__()

        self.up1 = Up(bottleneck_channels, features[3], features[3])  # 1024 + 512 -> 512
        self.up2 = Up(features[3], features[2], features[2])          # 512 + 256 -> 256
        self.up3 = Up(features[2], features[1], features[1])          # 256 + 128 -> 128
        self.up4 = Up(features[1], features[0], features[0])          # 128 + 64 -> 64

        self.seg_head = nn.Conv2d(features[0], out_classes, kernel_size=1)

    def forward(self, b, skips):
        s1, s2, s3, s4 = skips

        x = self.up1(b, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)

        out = self.seg_head(x)
        return out


class UNet(nn.Module):
    def __init__(self, n_classes=1, in_channels=3):
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(out_classes=n_classes)

    def forward(self, x):
        b, skips = self.encoder(x)
        out = self.decoder(b, skips)
        return out


if __name__ == "__main__":
    start = time.time()

    x = torch.randn(2, 3, 256, 256)
    model = UNet(n_classes=1)
    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    end = time.time()
    print(f"Spending time: {end - start:.4f} sec")