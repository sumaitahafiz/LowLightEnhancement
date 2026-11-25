import sys
from pathlib import Path

# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import torch
import matplotlib.pyplot as plt

from data.dataset import ExDarkDataset
from preprocessing.fusion import create_analysis_features
from preprocessing.illumination import estimate_illumination
from model import LightweightZeroReferenceCNN
from enhancement import AdaptiveEnhancement


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
# LOAD SAMPLE
# ============================================================

sample = dataset[0]

image = sample["image"]

image_path = sample["path"]


print("=" * 60)
print("ADAPTIVE ENHANCEMENT TEST")
print("=" * 60)

print("Image path          :", image_path)
print("Original image shape:", image.shape)
print("Image data type     :", image.dtype)
print("Image minimum       :", image.min().item())
print("Image maximum       :", image.max().item())


# ============================================================
# CREATE SIX-CHANNEL FEATURES
# ============================================================

features = create_analysis_features(image)


print()
print("Feature tensor shape:", features.shape)


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

rgb = image.unsqueeze(0)

features = features.unsqueeze(0)


# ============================================================
# ILLUMINATION MAP
# ============================================================

illumination = estimate_illumination(image)

illumination = illumination.unsqueeze(0)


print("RGB batch shape      :", rgb.shape)
print("Illumination shape   :", illumination.shape)


# ============================================================
# LIGHTWEIGHT CNN
# ============================================================

model = LightweightZeroReferenceCNN()

model.eval()


with torch.no_grad():

    parameters = model(features)


print()
print("CNN parameter shape  :", parameters.shape)
print(
    "Parameter minimum    :",
    parameters.min().item()
)
print(
    "Parameter maximum    :",
    parameters.max().item()
)


# ============================================================
# ADAPTIVE ENHANCEMENT
# ============================================================

enhancer = AdaptiveEnhancement()

enhancer.eval()


with torch.no_grad():

    enhanced = enhancer(
        rgb,
        illumination,
        parameters
    )


print()
print("Enhanced image shape :", enhanced.shape)
print(
    "Enhanced minimum     :",
    enhanced.min().item()
)
print(
    "Enhanced maximum     :",
    enhanced.max().item()
)


# ============================================================
# REMOVE BATCH DIMENSION FOR DISPLAY
# ============================================================

original_image = (
    rgb[0]
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)

illumination_image = (
    illumination[0, 0]
    .cpu()
    .numpy()
)

enhanced_image = (
    enhanced[0]
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)


# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


axes[0].imshow(original_image)

axes[0].set_title(
    "Original Low-Light Image"
)

axes[0].axis("off")


axes[1].imshow(
    illumination_image,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "Illumination Map"
)

axes[1].axis("off")


axes[2].imshow(enhanced_image)

axes[2].set_title(
    "Adaptive Enhancement"
)

axes[2].axis("off")


plt.tight_layout()

plt.show()