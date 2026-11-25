import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from model import LightweightZeroReferenceCNN
from preprocessing.fusion import create_analysis_features
from enhancement import AdaptiveEnhancement
from refinement import ImageRefinement

from losses.total_loss import TotalLoss


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "epoch_5.pth"
)

LOW_IMAGE_PATH = (
    Path(r"D:\12th Trimester\DIP\DIP Paper\Datasets")
    / "LOLdataset"
    / "eval15"
    / "low"
    / "1.png"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 256
DEVICE = torch.device("cpu")


# ============================================================
# HELPER
# ============================================================

def print_loss(name, value):
    print(f"{name:<15}: {value:.8f}")


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRAINED MODEL LOSS DIAGNOSIS")
print("=" * 70)

print()
print("Checkpoint:")
print(CHECKPOINT_PATH)

print()
print("Test image:")
print(LOW_IMAGE_PATH)

print()
print("Device:")
print(DEVICE)


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(LOW_IMAGE_PATH).convert("RGB")

image = image.resize(
    (IMAGE_SIZE, IMAGE_SIZE),
    Image.Resampling.BILINEAR
)

image_np = np.asarray(image).astype(np.float32) / 255.0

image_tensor = torch.from_numpy(
    image_np.transpose(2, 0, 1)
).float()

original = image_tensor.unsqueeze(0).to(DEVICE)


# ============================================================
# CREATE FEATURES
# ============================================================

features = create_analysis_features(
    image_tensor
)

features = features.unsqueeze(0).to(DEVICE)

illumination = features[:, 3:4, :, :]


print()
print("=" * 70)
print("PIPELINE SHAPES")
print("=" * 70)

print()
print(f"Original      : {original.shape}")
print(f"Features      : {features.shape}")
print(f"Illumination  : {illumination.shape}")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = LightweightZeroReferenceCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()


print()
print("Model loaded successfully.")


# ============================================================
# FORWARD PASS
# ============================================================

enhancement = AdaptiveEnhancement()
refinement = ImageRefinement()

with torch.no_grad():

    parameters = model(features)

    enhanced = enhancement(
        original,
        illumination,
        parameters
    )

    refined = refinement(
        original,
        enhanced
    )


# ============================================================
# LOSS OBJECT
# ============================================================

loss_function = TotalLoss()


# ============================================================
# LOSS ON ENHANCED IMAGE
# ============================================================

losses_enhanced = loss_function(
    original,
    enhanced,
    illumination
)


# ============================================================
# LOSS ON REFINED IMAGE
# ============================================================

losses_refined = loss_function(
    original,
    refined,
    illumination
)


# ============================================================
# PRINT ENHANCED LOSSES
# ============================================================

print()
print("=" * 70)
print("LOSS VALUES — ENHANCED IMAGE")
print("=" * 70)

print()

print_loss(
    "Illumination",
    losses_enhanced["illumination"].item()
)

print_loss(
    "Exposure",
    losses_enhanced["exposure"].item()
)

print_loss(
    "Spatial",
    losses_enhanced["spatial"].item()
)

print_loss(
    "Color",
    losses_enhanced["color"].item()
)

print_loss(
    "Noise",
    losses_enhanced["noise"].item()
)

print_loss(
    "Edge",
    losses_enhanced["edge"].item()
)

print_loss(
    "TOTAL",
    losses_enhanced["total"].item()
)


# ============================================================
# PRINT REFINED LOSSES
# ============================================================

print()
print("=" * 70)
print("LOSS VALUES — REFINED IMAGE")
print("=" * 70)

print()

print_loss(
    "Illumination",
    losses_refined["illumination"].item()
)

print_loss(
    "Exposure",
    losses_refined["exposure"].item()
)

print_loss(
    "Spatial",
    losses_refined["spatial"].item()
)

print_loss(
    "Color",
    losses_refined["color"].item()
)

print_loss(
    "Noise",
    losses_refined["noise"].item()
)

print_loss(
    "Edge",
    losses_refined["edge"].item()
)

print_loss(
    "TOTAL",
    losses_refined["total"].item()
)


# ============================================================
# LOSS DIFFERENCES
# ============================================================

print()
print("=" * 70)
print("REFINEMENT EFFECT ON LOSSES")
print("=" * 70)

print()

for name in [
    "illumination",
    "exposure",
    "spatial",
    "color",
    "noise",
    "edge",
    "total"
]:

    enhanced_value = losses_enhanced[name].item()
    refined_value = losses_refined[name].item()

    difference = refined_value - enhanced_value

    print(
        f"{name:<15}: "
        f"{difference:+.8f}"
    )


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 70)
print("LOSS DIAGNOSIS COMPLETED")
print("=" * 70)