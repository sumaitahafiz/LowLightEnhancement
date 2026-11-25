import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path

import matplotlib.pyplot as plt

from data.dataset import ExDarkDataset
from preprocessing.illumination import estimate_illumination


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
# ESTIMATE ILLUMINATION
# ============================================================

illumination = estimate_illumination(image)


# ============================================================
# CONVERT FOR DISPLAY
# ============================================================

rgb_image = image.permute(1, 2, 0).numpy()

illumination_image = illumination.squeeze(0).numpy()


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


# Illumination map
axes[1].imshow(
    illumination_image,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1].set_title("Estimated Illumination Map")
axes[1].axis("off")


plt.tight_layout()
plt.show()