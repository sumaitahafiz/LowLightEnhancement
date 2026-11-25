from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import ExDarkDataset


# ============================================================
# EXDARK DATASET PATH
# ============================================================

EXDARK_PATH = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\ExDark\ExDark"
)


# ============================================================
# LOAD DATASET
# ============================================================

dataset = ExDarkDataset(
    root_dir=EXDARK_PATH,
    image_size=256,
)


# ============================================================
# SELECT SOME IMAGES
# ============================================================

indices = [0, 100, 500, 1000]


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 10)
)

axes = axes.flatten()


# ============================================================
# DISPLAY IMAGES
# ============================================================

for ax, index in zip(axes, indices):

    sample = dataset[index]

    image = sample["image"]
    path = sample["path"]

    # PyTorch:
    # [C, H, W]
    #
    # Matplotlib:
    # [H, W, C]

    image = image.permute(1, 2, 0).numpy()

    ax.imshow(image)

    ax.set_title(
        Path(path).parent.name
    )

    ax.axis("off")


plt.tight_layout()

plt.show()