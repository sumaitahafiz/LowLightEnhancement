import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from data.dataset import ExDarkDataset
from preprocessing.edges import estimate_edges


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


# ============================================================
# EDGE ANALYSIS
# ============================================================

edges = estimate_edges(image)


# ============================================================
# DISPLAY
# ============================================================

rgb_image = image.permute(1, 2, 0).numpy()
edge_image = edges.squeeze(0).numpy()


fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


axes[0].imshow(rgb_image)
axes[0].set_title("Original Low-Light Image")
axes[0].axis("off")


axes[1].imshow(
    edge_image,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1].set_title("Sobel Edge / Detail Map")
axes[1].axis("off")


plt.tight_layout()
plt.show()