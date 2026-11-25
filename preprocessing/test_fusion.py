import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from data.dataset import ExDarkDataset
from preprocessing.fusion import create_analysis_features


# ============================================================
# DATASET
# ============================================================

EXDARK_PATH = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\ExDark\ExDark"
)

dataset = ExDarkDataset(
    root_dir=EXDARK_PATH,
    image_size=256,
)


# ============================================================
# LOAD IMAGE
# ============================================================

sample = dataset[0]

image = sample["image"]


# ============================================================
# CREATE FEATURES
# ============================================================

features = create_analysis_features(image)


# ============================================================
# CHECK SHAPE
# ============================================================

print("=" * 60)
print("ANALYSIS FEATURE TEST")
print("=" * 60)

print("Original image shape :", image.shape)
print("Feature tensor shape :", features.shape)

print()
print("Feature channels:")
print("  Channels 0-2 : RGB")
print("  Channel 3    : Illumination")
print("  Channel 4    : High-frequency variation")
print("  Channel 5    : Sobel edge/detail")


# ============================================================
# DISPLAY ANALYSIS MAPS
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)


axes[0, 0].imshow(
    features[0:3].permute(1, 2, 0).numpy()
)
axes[0, 0].set_title("RGB")
axes[0, 0].axis("off")


axes[0, 1].imshow(
    features[3].numpy(),
    cmap="gray",
    vmin=0,
    vmax=1
)
axes[0, 1].set_title("Illumination")
axes[0, 1].axis("off")


axes[0, 2].imshow(
    features[4].numpy(),
    cmap="gray",
    vmin=0,
    vmax=1
)
axes[0, 2].set_title("High-Frequency")
axes[0, 2].axis("off")


axes[1, 0].imshow(
    features[5].numpy(),
    cmap="gray",
    vmin=0,
    vmax=1
)
axes[1, 0].set_title("Sobel Detail")
axes[1, 0].axis("off")


# Hide unused panels
axes[1, 1].axis("off")
axes[1, 2].axis("off")


plt.tight_layout()
plt.show()