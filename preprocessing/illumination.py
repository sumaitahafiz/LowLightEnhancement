import torch


def estimate_illumination(image):
    """
    Estimate an illumination map from an RGB image.

    Parameters
    ----------
    image : torch.Tensor
        RGB image with shape [3, H, W].
        Values should be in the range [0, 1].

    Returns
    -------
    torch.Tensor
        Illumination map with shape [1, H, W].
    """

    if image.ndim != 3:
        raise ValueError(
            f"Expected image with shape [3, H, W], "
            f"but received shape {image.shape}"
        )

    if image.shape[0] != 3:
        raise ValueError(
            f"Expected 3 RGB channels, "
            f"but received {image.shape[0]}"
        )

    # Maximum RGB intensity at each pixel.
    illumination, _ = torch.max(
        image,
        dim=0,
        keepdim=True
    )

    return illumination