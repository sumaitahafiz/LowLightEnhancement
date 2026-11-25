import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageRefinement(nn.Module):
    """
    Refinement module for the adaptive-enhanced image.

    Components:
    1. Noise suppression
    2. Color correction
    3. Edge preservation
    """

    def __init__(self):
        super().__init__()

        # Fixed 5x5 Gaussian kernel for smooth/noise estimation
        gaussian_kernel = torch.tensor([
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ], dtype=torch.float32)

        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        # One kernel for each RGB channel
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        self.register_buffer("gaussian_kernel", gaussian_kernel)

    def gaussian_blur(self, image):
        """
        Apply Gaussian smoothing independently to RGB channels.
        """
        return F.conv2d(
            image,
            self.gaussian_kernel,
            padding=2,
            groups=3
        )

    def noise_suppression(self, image):
        """
        Suppress small-scale variations while preserving the
        overall image structure.
        """
        smoothed = self.gaussian_blur(image)

        # Residual between original and smoothed image
        detail = image - smoothed

        # Suppress a portion of high-frequency variation
        denoised = image - 0.35 * detail

        return torch.clamp(denoised, 0.0, 1.0)

    def color_correction(self, image):
        """
        Simple gray-world based color correction.

        Channel means are adjusted toward a common mean.
        """
        channel_mean = image.mean(dim=(2, 3), keepdim=True)

        target_mean = channel_mean.mean(
            dim=1,
            keepdim=True
        )

        gain = target_mean / (channel_mean + 1e-6)

        # Prevent excessive color amplification
        gain = torch.clamp(gain, 0.75, 1.25)

        corrected = image * gain

        return torch.clamp(corrected, 0.0, 1.0)

    def edge_preservation(self, original, image):
        """
        Preserve important structural details using
        an edge-aware detail component.
        """

        # Smoothed version of the refined image
        smoothed = self.gaussian_blur(image)

        # High-frequency detail
        detail = image - smoothed

        # Convert original image to grayscale
        gray = (
            0.299 * original[:, 0:1, :, :] +
            0.587 * original[:, 1:2, :, :] +
            0.114 * original[:, 2:3, :, :]
        )

        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=original.dtype,
            device=original.device
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            dtype=original.dtype,
            device=original.device
        ).view(1, 1, 3, 3)

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)

        edge_strength = torch.sqrt(
            grad_x ** 2 +
            grad_y ** 2 +
            1e-6
        )

        # Normalize edge strength
        edge_strength = edge_strength / (
            edge_strength.amax(
                dim=(2, 3),
                keepdim=True
            ) + 1e-6
        )

        # Preserve more detail around strong edges
        refined = image + 0.20 * edge_strength * detail

        return torch.clamp(refined, 0.0, 1.0)

    def forward(self, original, enhanced):
        """
        Full refinement pipeline.
        """

        # 1. Noise suppression
        denoised = self.noise_suppression(enhanced)

        # 2. Color correction
        color_corrected = self.color_correction(denoised)

        # 3. Edge preservation
        refined = self.edge_preservation(
            original,
            color_corrected
        )

        return refined


if __name__ == "__main__":

    print("=" * 60)
    print("IMAGE REFINEMENT MODULE TEST")
    print("=" * 60)

    # Dummy images for testing
    original = torch.rand(1, 3, 256, 256)
    enhanced = torch.rand(1, 3, 256, 256)

    refinement = ImageRefinement()

    refined = refinement(original, enhanced)

    print("Original shape :", original.shape)
    print("Enhanced shape :", enhanced.shape)
    print("Refined shape  :", refined.shape)

    print()
    print("Refined minimum:", refined.min().item())
    print("Refined maximum:", refined.max().item())

    print("=" * 60)
    print("REFINEMENT TEST COMPLETED")
    print("=" * 60)