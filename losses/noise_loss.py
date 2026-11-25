import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, original, enhanced):
        # Convert images to grayscale
        original_gray = (
            0.299 * original[:, 0:1, :, :] +
            0.587 * original[:, 1:2, :, :] +
            0.114 * original[:, 2:3, :, :]
        )

        enhanced_gray = (
            0.299 * enhanced[:, 0:1, :, :] +
            0.587 * enhanced[:, 1:2, :, :] +
            0.114 * enhanced[:, 2:3, :, :]
        )

        # Estimate smooth background using average pooling
        original_smooth = F.avg_pool2d(
            original_gray,
            kernel_size=3,
            stride=1,
            padding=1
        )

        enhanced_smooth = F.avg_pool2d(
            enhanced_gray,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # High-frequency residual
        original_noise = original_gray - original_smooth
        enhanced_noise = enhanced_gray - enhanced_smooth

        # Penalize excessive high-frequency energy
        original_energy = torch.mean(torch.abs(original_noise))
        enhanced_energy = torch.mean(torch.abs(enhanced_noise))

        excess_noise = torch.relu(
            enhanced_energy - original_energy
        )

        return excess_noise


if __name__ == "__main__":

    print("=" * 60)
    print("NOISE LOSS TEST")
    print("=" * 60)

    original = torch.rand(1, 3, 256, 256)
    enhanced = torch.rand(1, 3, 256, 256)

    loss_function = NoiseLoss()

    loss = loss_function(
        original,
        enhanced
    )

    print("Original shape :", original.shape)
    print("Enhanced shape :", enhanced.shape)
    print("Loss value     :", loss.item())

    print("=" * 60)
    print("NOISE LOSS TEST COMPLETED")
    print("=" * 60)