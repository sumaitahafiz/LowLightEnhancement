import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from data.dataset import ExDarkDataset
from preprocessing.fusion import create_analysis_features
from model import LightweightZeroReferenceCNN
from enhancement import AdaptiveEnhancement
from refinement import ImageRefinement
from losses.total_loss import TotalLoss


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


print("=" * 60)
print("TOTAL LOSS REAL PIPELINE TEST")
print("=" * 60)


# ============================================================
# LOAD IMAGE
# ============================================================

sample = dataset[0]

image = sample["image"]
image_path = sample["path"]

print("Image path          :", image_path)
print("Original image shape:", image.shape)


# ============================================================
# FEATURE FUSION
# ============================================================

features = create_analysis_features(image)

print("Feature tensor shape:", features.shape)


# ============================================================
# SEPARATE RGB + ILLUMINATION
# ============================================================

rgb = features[0:3].unsqueeze(0)
illumination = features[3:4].unsqueeze(0)

print("RGB batch shape     :", rgb.shape)
print("Illumination shape  :", illumination.shape)


# ============================================================
# CNN
# ============================================================

model = LightweightZeroReferenceCNN()

parameters = model(features.unsqueeze(0))

print("CNN parameter shape :", parameters.shape)


# ============================================================
# ADAPTIVE ENHANCEMENT
# ============================================================

enhancement = AdaptiveEnhancement()

enhanced = enhancement(
    rgb,
    illumination,
    parameters
)

print("Enhanced shape      :", enhanced.shape)


# ============================================================
# REFINEMENT
# ============================================================

refinement = ImageRefinement()

refined = refinement(
    rgb,
    enhanced
)

print("Refined shape       :", refined.shape)


# ============================================================
# TOTAL LOSS
# ============================================================

total_loss_function = TotalLoss()

losses = total_loss_function(
    rgb,
    refined,
    illumination
)


# ============================================================
# PRINT INDIVIDUAL LOSSES
# ============================================================

print()
print("Individual losses:")

print(
    "  Illumination :",
    losses["illumination"].item()
)

print(
    "  Exposure     :",
    losses["exposure"].item()
)

print(
    "  Spatial      :",
    losses["spatial"].item()
)

print(
    "  Color        :",
    losses["color"].item()
)

print(
    "  Noise        :",
    losses["noise"].item()
)

print(
    "  Edge         :",
    losses["edge"].item()
)

print()
print(
    "Total loss     :",
    losses["total"].item()
)


# ============================================================
# BACKPROPAGATION
# ============================================================

losses["total"].backward()

gradient_exists = any(
    parameter.grad is not None
    for parameter in model.parameters()
)

print()
print("CNN gradient available :", gradient_exists)


print("=" * 60)
print("TOTAL LOSS REAL PIPELINE TEST COMPLETED")
print("=" * 60)