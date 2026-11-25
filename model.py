import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Lightweight convolutional block used for feature extraction.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class LightweightZeroReferenceCNN(nn.Module):
    """
    Lightweight CNN for zero-reference low-light enhancement.

    Input:
        6-channel tensor:
        RGB (3) +
        Illumination (1) +
        High-Frequency (1) +
        Sobel Detail (1)

    Output:
        3-channel enhancement parameter map.
    """

    def __init__(self):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            ConvBlock(6, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 16)
        )

        # Enhancement parameter prediction
        self.parameter_head = nn.Sequential(
            nn.Conv2d(
                16,
                3,
                kernel_size=3,
                padding=1,
                bias=True
            ),
        )

    def forward(self, x):
        features = self.features(x)

        enhancement_parameters = self.parameter_head(features)

        return enhancement_parameters


if __name__ == "__main__":
    # Simple standalone model test

    model = LightweightZeroReferenceCNN()

    dummy_input = torch.randn(1, 6, 256, 256)

    output = model(dummy_input)

    print("=" * 60)
    print("Lightweight Zero-Reference CNN Test")
    print("=" * 60)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Output min  :", output.min().item())
    print("Output max  :", output.max().item())

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Parameters  :", total_parameters)
    print("=" * 60)