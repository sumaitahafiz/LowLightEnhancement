import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from losses.illumination_loss import IlluminationLoss
from losses.exposure_loss import ExposureLoss
from losses.spatial_loss import SpatialLoss
from losses.color_loss import ColorLoss
from losses.noise_loss import NoiseLoss
from losses.edge_loss import EdgeLoss


class TotalLoss(nn.Module):
    def __init__(
        self,
        lambda_illumination=1.0,
        lambda_exposure=1.0,
        lambda_spatial=1.0,
        lambda_color=1.0,
        lambda_noise=1.0,
        lambda_edge=1.0,
    ):
        super().__init__()

        # Loss functions
        self.illumination_loss = IlluminationLoss()
        self.exposure_loss = ExposureLoss()
        self.spatial_loss = SpatialLoss()
        self.color_loss = ColorLoss()
        self.noise_loss = NoiseLoss()
        self.edge_loss = EdgeLoss()

        # Loss weights
        self.lambda_illumination = lambda_illumination
        self.lambda_exposure = lambda_exposure
        self.lambda_spatial = lambda_spatial
        self.lambda_color = lambda_color
        self.lambda_noise = lambda_noise
        self.lambda_edge = lambda_edge

    def forward(self, original, enhanced, illumination):

        # Individual losses
        illumination_loss = self.illumination_loss(
            enhanced,
            illumination
        )

        exposure_loss = self.exposure_loss(
            enhanced
        )

        spatial_loss = self.spatial_loss(
            original,
            enhanced
        )

        color_loss = self.color_loss(
            original,
            enhanced
        )

        noise_loss = self.noise_loss(
            original,
            enhanced
        )

        edge_loss = self.edge_loss(
            original,
            enhanced
        )

        # Weighted total loss
        total_loss = (
            self.lambda_illumination * illumination_loss
            + self.lambda_exposure * exposure_loss
            + self.lambda_spatial * spatial_loss
            + self.lambda_color * color_loss
            + self.lambda_noise * noise_loss
            + self.lambda_edge * edge_loss
        )

        return {
            "total": total_loss,
            "illumination": illumination_loss,
            "exposure": exposure_loss,
            "spatial": spatial_loss,
            "color": color_loss,
            "noise": noise_loss,
            "edge": edge_loss,
        }


if __name__ == "__main__":

    print("=" * 60)
    print("TOTAL LOSS TEST")
    print("=" * 60)

    original = torch.rand(1, 3, 256, 256)
    enhanced = torch.rand(1, 3, 256, 256)
    illumination = torch.rand(1, 1, 256, 256)

    loss_function = TotalLoss()

    losses = loss_function(
        original,
        enhanced,
        illumination
    )

    print("Original shape     :", original.shape)
    print("Enhanced shape     :", enhanced.shape)
    print("Illumination shape :", illumination.shape)

    print()
    print("Individual losses:")
    print("  Illumination :", losses["illumination"].item())
    print("  Exposure     :", losses["exposure"].item())
    print("  Spatial      :", losses["spatial"].item())
    print("  Color        :", losses["color"].item())
    print("  Noise        :", losses["noise"].item())
    print("  Edge         :", losses["edge"].item())

    print()
    print("Total loss     :", losses["total"].item())

    print("=" * 60)
    print("TOTAL LOSS TEST COMPLETED")
    print("=" * 60)