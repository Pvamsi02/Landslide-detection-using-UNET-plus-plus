import torch
from torch import nn
import torch.nn.functional as F

class NestedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NestedConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, n_classes, n_channels=14):
        super(UNetPlusPlus, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv0_0 = NestedConvBlock(n_channels, 64)
        self.conv1_0 = NestedConvBlock(64, 128)
        self.conv2_0 = NestedConvBlock(128, 256)
        self.conv3_0 = NestedConvBlock(256, 512)
        self.conv4_0 = NestedConvBlock(512, 1024)

        self.conv0_1 = NestedConvBlock(64 + 128, 64)
        self.conv1_1 = NestedConvBlock(128 + 256, 128)
        self.conv2_1 = NestedConvBlock(256 + 512, 256)
        self.conv3_1 = NestedConvBlock(512 + 1024, 512)

        self.conv0_2 = NestedConvBlock(64*2 + 128, 64)
        self.conv1_2 = NestedConvBlock(128*2 + 256, 128)
        self.conv2_2 = NestedConvBlock(256*2 + 512, 256)

        self.conv0_3 = NestedConvBlock(64*3 + 128, 64)
        self.conv1_3 = NestedConvBlock(128*3 + 256, 128)

        self.conv0_4 = NestedConvBlock(64*4 + 128, 64)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(F.max_pool2d(x0_0, 2))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(F.max_pool2d(x1_0, 2))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(F.max_pool2d(x2_0, 2))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(F.max_pool2d(x3_0, 2))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        return self.final_conv(x0_4)
