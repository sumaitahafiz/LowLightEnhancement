import torch
import torch.nn as nn


class SpatialLoss(nn.Module):
    """
    Spatial consistency loss.

    Encourages the enhanced image to preserve the
    spatial relationships and local gradients of the
    original low-light image.
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
            Refined/enhanced RGB image [B, 3, H, W]

        Returns
        -------
        torch.Tensor
            Scalar spatial loss.
        """

        # ----------------------------------------------------
        # Convert both images to grayscale
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Horizontal gradients
        # ----------------------------------------------------

        original_dx = (
            original_gray[:, :, :, 1:] -
            original_gray[:, :, :, :-1]
        )

        enhanced_dx = (
            enhanced_gray[:, :, :, 1:] -
            enhanced_gray[:, :, :, :-1]
        )

        # ----------------------------------------------------
        # Vertical gradients
        # ----------------------------------------------------

        original_dy = (
            original_gray[:, :, 1:, :] -
            original_gray[:, :, :-1, :]
        )

        enhanced_dy = (
            enhanced_gray[:, :, 1:, :] -
            enhanced_gray[:, :, :-1, :]
        )

        # ----------------------------------------------------
        # Compare spatial gradients
        # ----------------------------------------------------

        loss_x = torch.mean(
            torch.abs(
                enhanced_dx - original_dx
            )
        )

        loss_y = torch.mean(
            torch.abs(
                enhanced_dy - original_dy
            )
        )

        # ----------------------------------------------------
        # Total spatial loss
        # ----------------------------------------------------

        loss = loss_x + loss_y

        return loss


if __name__ == "__main__":

    print("=" * 60)
    print("SPATIAL LOSS TEST")
    print("=" * 60)

    original = torch.rand(
        1, 3, 256, 256
    )

    enhanced = torch.rand(
        1, 3, 256, 256
    )

    loss_function = SpatialLoss()

    loss = loss_function(
        original,
        enhanced
    )

    print("Original shape :", original.shape)
    print("Enhanced shape :", enhanced.shape)
    print("Loss value     :", loss.item())

    print("=" * 60)
    print("SPATIAL LOSS TEST COMPLETED")
    print("=" * 60)