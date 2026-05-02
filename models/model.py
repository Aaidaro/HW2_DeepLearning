import torch.nn as nn

from models.blocks import (
    ConvBNReLU,
    ResidualBlockA,
    InceptionBlockB,
    ResNeXtBlockC
)


class PersianMNISTCNN(nn.Module):
    """
    Full CNN architecture for Persian MNIST.

    block_type:
        "A" -> Residual Block
        "B" -> Inception-like Block
        "C" -> ResNeXt-like Block
    """

    def __init__(
        self,
        num_classes=10,
        block_type="A",
        dropout_rate=0.2,
        groups=8,
        bottleneck_ratio=0.5
    ):
        super().__init__()

        self.block_type = block_type

        def make_block(in_channels, out_channels):
            if block_type == "A":
                return ResidualBlockA(in_channels, out_channels)

            if block_type == "B":
                return InceptionBlockB(in_channels, out_channels)

            if block_type == "C":
                return ResNeXtBlockC(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=groups,
                    bottleneck_ratio=bottleneck_ratio
                )

            raise ValueError("block_type must be one of: A, B, C")

        self.features = nn.Sequential(
            # 1) Conv 3x3: 1 -> 32
            ConvBNReLU(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            # 2) Conv 3x3: 32 -> 64 + MaxPool
            ConvBNReLU(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3) Special block: 64 -> 64
            make_block(64, 64),

            # 4) Special block: 64 -> 128 + MaxPool
            make_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 5) Conv 3x3: 128 -> 256 + MaxPool
            ConvBNReLU(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 6) Special block: 256 -> 256
            make_block(256, 256),

            # Average Pool
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Do not use Softmax here.
        # CrossEntropyLoss expects raw logits.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x