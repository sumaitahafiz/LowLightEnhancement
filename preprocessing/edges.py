import torch
import torch.nn.functional as F


def estimate_edges(image):
    """
    Estimate structural edges using Sobel gradients.

    Parameters
    ----------
    image : torch.Tensor
        RGB image with shape [3, H, W].
        Values should be in the range [0, 1].

    Returns
    -------
    torch.Tensor
        Edge/detail map with shape [1, H, W].
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

    # --------------------------------------------------------
    # Convert RGB to grayscale
    # --------------------------------------------------------

    gray = (
        0.299 * image[0]
        + 0.587 * image[1]
        + 0.114 * image[2]
    )

    # [H, W] -> [1, 1, H, W]
    gray = gray.unsqueeze(0).unsqueeze(0)

    # --------------------------------------------------------
    # Sobel kernels
    # --------------------------------------------------------

    sobel_x = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)

    # --------------------------------------------------------
    # Horizontal and vertical gradients
    # --------------------------------------------------------

    gradient_x = F.conv2d(
        gray,
        sobel_x,
        padding=1
    )

    gradient_y = F.conv2d(
        gray,
        sobel_y,
        padding=1
    )

    # --------------------------------------------------------
    # Gradient magnitude
    # --------------------------------------------------------

    edge_map = torch.sqrt(
        gradient_x ** 2
        + gradient_y ** 2
        + 1e-8
    )

    # Remove batch dimension
    edge_map = edge_map.squeeze(0)

    # Normalize to [0, 1]
    min_value = edge_map.min()
    max_value = edge_map.max()

    if max_value > min_value:
        edge_map = (
            edge_map - min_value
        ) / (
            max_value - min_value
        )

    return edge_map