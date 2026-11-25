import torch
import torch.nn as nn


class ColorLoss(nn.Module):
    """
    Color consistency loss.

    Encourages the enhanced image to avoid introducing
    excessive color shifts relative to the original image.
    """

    def __init__(self):
        super().__init__()

    def forward(self, original, enhanced):
        """
        Parameters
        ----------
        original : torch.Tensor
            Original RGB image [B, 3, H, W]

        enhanced : torch.Tensor
            Enhanced/refined RGB image [B, 3, H, W]

        Returns
        -------
        torch.Tensor
            Scalar color loss.
        """

        # ----------------------------------------------------
        # Calculate mean RGB values
        # ----------------------------------------------------

        original_mean = original.mean(
            dim=(2, 3),
            keepdim=True
        )

        enhanced_mean = enhanced.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Normalize each image's color proportions
        # ----------------------------------------------------

        original_sum = (
            original_mean.sum(
                dim=1,
                keepdim=True
            ) + 1e-6
        )

        enhanced_sum = (
            enhanced_mean.sum(
                dim=1,
                keepdim=True
            ) + 1e-6
        )

        original_ratio = (
            original_mean / original_sum
        )

        enhanced_ratio = (
            enhanced_mean / enhanced_sum
        )

        # ----------------------------------------------------
        # Compare color proportions
        # ----------------------------------------------------

        loss = torch.mean(
            torch.abs(
                enhanced_ratio -
                original_ratio
            )
        )

        return loss


if __name__ == "__main__":

    print("=" * 60)
    print("COLOR LOSS TEST")
    print("=" * 60)

    original = torch.rand(
        1, 3, 256, 256
    )

    enhanced = torch.rand(
        1, 3, 256, 256
    )

    loss_function = ColorLoss()

    loss = loss_function(
        original,
        enhanced
    )

    print("Original shape :", original.shape)
    print("Enhanced shape :", enhanced.shape)
    print("Loss value     :", loss.item())

    print("=" * 60)
    print("COLOR LOSS TEST COMPLETED")
    print("=" * 60)