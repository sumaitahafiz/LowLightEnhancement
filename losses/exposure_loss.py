import torch
import torch.nn as nn
import torch.nn.functional as F


class ExposureLoss(nn.Module):
    """
    Contrast / exposure loss.

    Encourages local image regions to approach a target
    exposure level without requiring a ground-truth image.
    """

    def __init__(self, target_mean=0.6, patch_size=16):
        super().__init__()

        self.target_mean = target_mean
        self.patch_size = patch_size

    def forward(self, image):
        """
        Parameters
        ----------
        image : torch.Tensor
            RGB image with shape [B, 3, H, W]

        Returns
        -------
        torch.Tensor
            Scalar exposure loss.
        """

        # ----------------------------------------------------
        # Convert RGB image to grayscale
        # ----------------------------------------------------

        gray = (
            0.299 * image[:, 0:1, :, :] +
            0.587 * image[:, 1:2, :, :] +
            0.114 * image[:, 2:3, :, :]
        )

        # ----------------------------------------------------
        # Calculate local average exposure
        # ----------------------------------------------------

        local_mean = F.avg_pool2d(
            gray,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # ----------------------------------------------------
        # Compare local exposure against target
        # ----------------------------------------------------

        target = torch.full_like(
            local_mean,
            self.target_mean
        )

        loss = torch.mean(
            (local_mean - target) ** 2
        )

        return loss


if __name__ == "__main__":

    print("=" * 60)
    print("CONTRAST / EXPOSURE LOSS TEST")
    print("=" * 60)

    image = torch.rand(
        1, 3, 256, 256
    )

    loss_function = ExposureLoss()

    loss = loss_function(image)

    print("Image shape :", image.shape)
    print("Loss value  :", loss.item())

    print("=" * 60)
    print("EXPOSURE LOSS TEST COMPLETED")
    print("=" * 60)