import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from data.dataset import ExDarkDataset
from preprocessing.noise import estimate_noise


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
# SELECT IMAGE
# ============================================================

sample = dataset[0]

image = sample["image"]
image_path = sample["path"]


# ============================================================
# ESTIMATE NOISE
# ============================================================

noise = estimate_noise(
    image,
    kernel_size=5
)


# ============================================================
# CONVERT FOR DISPLAY
# ============================================================

rgb_image = image.permute(1, 2, 0).numpy()

noise_image = noise.squeeze(0).numpy()


# ============================================================
# DISPLAY
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


# Original
axes[0].imshow(rgb_image)
axes[0].set_title("Original Low-Light Image")
axes[0].axis("off")


# Noise map
axes[1].imshow(
    noise_image,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1].set_title("Estimated Noise / High-Frequency Map")
axes[1].axis("off")


plt.tight_layout()
plt.show()