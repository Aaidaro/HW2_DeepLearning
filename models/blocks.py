import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlockA(nn.Module):
    """
    Block A:
    Residual block.

    If input and output channels are the same:
        shortcut = identity

    If channels are different:
        shortcut = 1x1 convolution
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.main_path = nn.Sequential(
            ConvBNReLU(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main_path(x)
        shortcut = self.shortcut(x)
        out = out + shortcut
        out = self.relu(out)
        return out


class InceptionBlockB(nn.Module):
    """
    Block B:
    Inception-like block.

    Four parallel branches:
        1) AvgPool -> 1x1 Conv
        2) 1x1 Conv
        3) 1x1 Conv -> 3x3 Conv
        4) 1x1 Conv -> 3x3 Conv -> 3x3 Conv

    Outputs are concatenated along channel dimension.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        if out_channels % 4 != 0:
            raise ValueError("For Block B, out_channels must be divisible by 4.")

        branch_channels = out_channels // 4

        self.branch1 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ConvBNReLU(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=1,
                padding=0
            )
        )

        self.branch2 = ConvBNReLU(
            in_channels=in_channels,
            out_channels=branch_channels,
            kernel_size=1,
            padding=0
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=1,
                padding=0
            ),
            ConvBNReLU(
                in_channels=branch_channels,
                out_channels=branch_channels,
                kernel_size=3,
                padding=1
            )
        )

        self.branch4 = nn.Sequential(
            ConvBNReLU(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=1,
                padding=0
            ),
            ConvBNReLU(
                in_channels=branch_channels,
                out_channels=branch_channels,
                kernel_size=3,
                padding=1
            ),
            ConvBNReLU(
                in_channels=branch_channels,
                out_channels=branch_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class ResNeXtBlockC(nn.Module):
    """
    Block C:
    ResNeXt-like block.

    Structure:
        1x1 Conv -> 3x3 Grouped Conv -> 1x1 Conv
        + residual shortcut
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        groups=8 ,
        bottleneck_ratio=0.5
    ):
        super().__init__()

        bottleneck_channels = int(out_channels * bottleneck_ratio)

        if bottleneck_channels < groups:
            bottleneck_channels = groups

        if bottleneck_channels % groups != 0:
            bottleneck_channels = groups * ((bottleneck_channels // groups) + 1)

        self.main_path = nn.Sequential(
            ConvBNReLU(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                padding=0
            ),
            ConvBNReLU(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                kernel_size=3,
                padding=1,
                groups=groups
            ),
            nn.Conv2d(
                in_channels=bottleneck_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main_path(x)
        shortcut = self.shortcut(x)
        out = out + shortcut
        out = self.relu(out)
        return out