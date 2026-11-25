import torch

from preprocessing.illumination import estimate_illumination
from preprocessing.noise import estimate_noise
from preprocessing.edges import estimate_edges


def create_analysis_features(image):
    """
    Create a six-channel feature representation.

    Channels:
        0-2 : RGB image
        3   : Illumination map
        4   : High-frequency variation map
        5   : Sobel edge/detail map

    Input:
        [3, H, W]

    Output:
        [6, H, W]
    """

    if image.ndim != 3:
        raise ValueError(
            f"Expected [3, H, W], got {image.shape}"
        )

    illumination = estimate_illumination(image)

    high_frequency = estimate_noise(image)

    edges = estimate_edges(image)

    features = torch.cat(
        [
            image,
            illumination,
            high_frequency,
            edges,
        ],
        dim=0
    )

    return features