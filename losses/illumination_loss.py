import torch
import torch.nn as nn


class IlluminationLoss(nn.Module):
    """
    Edge-aware illumination smoothness loss.

    The loss encourages the illumination map to vary smoothly
    while allowing changes around strong image edges.
    """

    def __init__(self, edge_weight=10.0):
        super().__init__()

        self.edge_weight = edge_weight

    def forward(self, image, illumination):
        """
        Parameters
        ----------
        image : torch.Tensor
            RGB image with shape [B, 3, H, W]

        illumination : torch.Tensor
            Illumination map with shape [B, 1, H, W]

        Returns
        -------
        torch.Tensor
            Scalar illumination loss.
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
        # Image gradients
        # ----------------------------------------------------

        gray_dx = torch.abs(
            gray[:, :, :, 1:] -
            gray[:, :, :, :-1]
        )

        gray_dy = torch.abs(
            gray[:, :, 1:, :] -
            gray[:, :, :-1, :]
        )

        # ----------------------------------------------------
        # Illumination gradients
        # ----------------------------------------------------

        illumination_dx = torch.abs(
            illumination[:, :, :, 1:] -
            illumination[:, :, :, :-1]
        )

        illumination_dy = torch.abs(
            illumination[:, :, 1:, :] -
            illumination[:, :, :-1, :]
        )

        # ----------------------------------------------------
        # Edge-aware weighting
        #
        # Strong image edges receive smaller smoothness
        # penalties.
        # ----------------------------------------------------

        weight_x = torch.exp(
            -self.edge_weight * gray_dx
        )

        weight_y = torch.exp(
            -self.edge_weight * gray_dy
        )

        # ----------------------------------------------------
        # Weighted illumination smoothness
        # ----------------------------------------------------

        loss_x = (
            weight_x * illumination_dx
        ).mean()

        loss_y = (
            weight_y * illumination_dy
        ).mean()

        # ----------------------------------------------------
        # Final illumination loss
        # ----------------------------------------------------

        loss = loss_x + loss_y

        return loss


if __name__ == "__main__":

    print("=" * 60)
    print("ILLUMINATION LOSS TEST")
    print("=" * 60)

    # Dummy image
    image = torch.rand(1, 3, 256, 256)

    # Dummy illumination map
    illumination = torch.rand(1, 1, 256, 256)

    loss_function = IlluminationLoss()

    loss = loss_function(
        image,
        illumination
    )

    print("Image shape        :", image.shape)
    print("Illumination shape :", illumination.shape)
    print("Loss value         :", loss.item())

    print("=" * 60)
    print("ILLUMINATION LOSS TEST COMPLETED")
    print("=" * 60)