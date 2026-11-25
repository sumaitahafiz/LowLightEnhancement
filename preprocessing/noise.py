import torch
import torch.nn.functional as F


def estimate_noise(image, kernel_size=5):
    """
    Estimate local high-frequency variation.

    Parameters
    ----------
    image : torch.Tensor
        RGB image with shape [3, H, W].
        Values should be in the range [0, 1].

    kernel_size : int
        Size of the local mean filter.

    Returns
    -------
    torch.Tensor
        Noise/activity map with shape [1, H, W].
    """

    if image.ndim != 3:
        raise ValueError(
            f"Expected image with shape [3, H, W], "
            f"but received {image.shape}"
        )

    if image.shape[0] != 3:
        raise ValueError(
            f"Expected 3 RGB channels, "
            f"but received {image.shape[0]}"
        )

    if kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be an odd number."
        )

    # --------------------------------------------------------
    # Convert RGB image to grayscale
    # --------------------------------------------------------

    gray = (
        0.299 * image[0]
        + 0.587 * image[1]
        + 0.114 * image[2]
    )

    # Add batch and channel dimensions:
    #
    # [H, W]
    #   ↓
    # [1, 1, H, W]

    gray = gray.unsqueeze(0).unsqueeze(0)

    # --------------------------------------------------------
    # Local mean filtering
    # --------------------------------------------------------

    padding = kernel_size // 2

    local_mean = F.avg_pool2d(
        gray,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    # --------------------------------------------------------
    # Local high-frequency variation
    # --------------------------------------------------------

    noise_map = torch.abs(
        gray - local_mean
    )

    # Remove batch/channel dimensions
    #
    # [1, 1, H, W]
    #      ↓
    # [1, H, W]

    noise_map = noise_map.squeeze(0)

    # Normalize to [0, 1]
    min_value = noise_map.min()
    max_value = noise_map.max()

    if max_value > min_value:
        noise_map = (
            noise_map - min_value
        ) / (
            max_value - min_value
        )

    return noise_map