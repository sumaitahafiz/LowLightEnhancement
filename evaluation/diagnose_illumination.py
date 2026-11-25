import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from data.dataset import ExDarkDataset
from preprocessing.illumination import estimate_illumination


# ============================================================
# PATHS
# ============================================================

LOL_V1_PATH = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\LOLdataset\eval15"
)

LOW_PATH = LOL_V1_PATH / "low"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 256


# ============================================================
# DATASET
# ============================================================

dataset = ExDarkDataset(
    root_dir=LOW_PATH,
    image_size=IMAGE_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# ============================================================
# ACCUMULATORS
# ============================================================

total_pixels = 0

red_wins = 0
green_wins = 0
blue_wins = 0

illumination_sum = 0.0
illumination_sq_sum = 0.0

rgb_sum = torch.zeros(3)

# Count ties separately.
tie_pixels = 0


# ============================================================
# ANALYSIS
# ============================================================

for batch in loader:

    image = batch["image"]

    # [1, 3, H, W] -> [3, H, W]
    image = image[0]

    red = image[0]
    green = image[1]
    blue = image[2]

    # --------------------------------------------------------
    # Illumination map using the actual implementation
    # --------------------------------------------------------

    illumination = estimate_illumination(image)

    # Remove channel dimension
    illumination_2d = illumination[0]

    # --------------------------------------------------------
    # Determine which RGB channel produces illumination
    # --------------------------------------------------------

    rgb = torch.stack(
        [red, green, blue],
        dim=0
    )

    max_values, max_indices = torch.max(
        rgb,
        dim=0
    )

    # --------------------------------------------------------
    # Count winners
    # --------------------------------------------------------

    red_wins += torch.sum(max_indices == 0).item()
    green_wins += torch.sum(max_indices == 1).item()
    blue_wins += torch.sum(max_indices == 2).item()

    # --------------------------------------------------------
    # Detect ties
    # --------------------------------------------------------

    red_is_max = red == max_values
    green_is_max = green == max_values
    blue_is_max = blue == max_values

    number_of_max_channels = (
        red_is_max.int()
        + green_is_max.int()
        + blue_is_max.int()
    )

    tie_pixels += torch.sum(
        number_of_max_channels > 1
    ).item()

    # --------------------------------------------------------
    # Illumination statistics
    # --------------------------------------------------------

    illumination_sum += illumination_2d.sum().item()
    illumination_sq_sum += (
        illumination_2d ** 2
    ).sum().item()

    # --------------------------------------------------------
    # RGB statistics
    # --------------------------------------------------------

    rgb_sum += rgb.reshape(3, -1).sum(dim=1)

    total_pixels += red.numel()


# ============================================================
# FINAL STATISTICS
# ============================================================

illumination_mean = (
    illumination_sum / total_pixels
)

illumination_variance = (
    illumination_sq_sum / total_pixels
    - illumination_mean ** 2
)

illumination_std = illumination_variance ** 0.5

rgb_mean = rgb_sum / total_pixels


red_percentage = (
    100.0 * red_wins / total_pixels
)

green_percentage = (
    100.0 * green_wins / total_pixels
)

blue_percentage = (
    100.0 * blue_wins / total_pixels
)

tie_percentage = (
    100.0 * tie_pixels / total_pixels
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 60)
print("ILLUMINATION MAP DIAGNOSTIC")
print("=" * 60)

print()
print("NUMBER OF IMAGES")
print(len(dataset))

print()
print("AVERAGE RGB CHANNEL MEANS")

print(
    f"R : {rgb_mean[0].item():.8f}"
)

print(
    f"G : {rgb_mean[1].item():.8f}"
)

print(
    f"B : {rgb_mean[2].item():.8f}"
)

print()
print("ILLUMINATION MAP STATISTICS")

print(
    f"Mean : {illumination_mean:.8f}"
)

print(
    f"Std  : {illumination_std:.8f}"
)

print()
print("CHANNEL PRODUCING MAXIMUM ILLUMINATION")

print(
    f"Red   : {red_percentage:.2f}%"
)

print(
    f"Green : {green_percentage:.2f}%"
)

print(
    f"Blue  : {blue_percentage:.2f}%"
)

print()
print("TIE PIXELS")

print(
    f"{tie_percentage:.2f}%"
)

print()
print("=" * 60)