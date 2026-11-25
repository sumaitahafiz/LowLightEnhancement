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
# HELPER FUNCTION
# ============================================================

def compute_gradient_norm(loss, model):
    """
    Compute the total L2 gradient norm of the CNN parameters
    for one individual loss term.
    """

    model.zero_grad(set_to_none=True)

    loss.backward(
        retain_graph=True
    )

    total_squared_norm = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_norm = (
                parameter.grad.detach().norm(2).item()
            )

            total_squared_norm += (
                gradient_norm ** 2
            )

    total_norm = total_squared_norm ** 0.5

    model.zero_grad(set_to_none=True)

    return total_norm


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("LOSS GRADIENT DIAGNOSIS")
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

image = Image.open(
    LOW_IMAGE_PATH
).convert("RGB")

image = image.resize(
    (IMAGE_SIZE, IMAGE_SIZE),
    Image.Resampling.BILINEAR
)

image_np = (
    np.asarray(image)
    .astype(np.float32)
    / 255.0
)

image_tensor = torch.from_numpy(
    image_np.transpose(2, 0, 1)
).float()

original = (
    image_tensor
    .unsqueeze(0)
    .to(DEVICE)
)


# ============================================================
# CREATE ANALYSIS FEATURES
# ============================================================

features = create_analysis_features(
    image_tensor
)

features = (
    features
    .unsqueeze(0)
    .to(DEVICE)
)

rgb = features[:, 0:3, :, :]

illumination = features[:, 3:4, :, :]


# ============================================================
# PRINT SHAPES
# ============================================================

print()
print("=" * 70)
print("PIPELINE SHAPES")
print("=" * 70)

print()
print(f"Original      : {original.shape}")
print(f"Features      : {features.shape}")
print(f"RGB           : {rgb.shape}")
print(f"Illumination  : {illumination.shape}")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = LightweightZeroReferenceCNN().to(
    DEVICE
)

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

    model.load_state_dict(
        checkpoint
    )

model.train()


print()
print("Model loaded successfully.")


# ============================================================
# ENHANCEMENT + REFINEMENT
# ============================================================

enhancement = AdaptiveEnhancement().to(
    DEVICE
)

refinement = ImageRefinement().to(
    DEVICE
)

parameters = model(features)

enhanced = enhancement(
    rgb,
    illumination,
    parameters
)

refined = refinement(
    rgb,
    enhanced
)


# ============================================================
# LOSS FUNCTION
# ============================================================

loss_function = TotalLoss().to(
    DEVICE
)


# ============================================================
# COMPUTE LOSSES
# ============================================================

losses = loss_function(
    rgb,
    refined,
    illumination
)


# ============================================================
# PRINT LOSS VALUES
# ============================================================

print()
print("=" * 70)
print("LOSS VALUES USED FOR GRADIENT ANALYSIS")
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

    print(
        f"{name:<15}: "
        f"{losses[name].item():.8f}"
    )


# ============================================================
# GRADIENT ANALYSIS
# ============================================================

print()
print("=" * 70)
print("CNN GRADIENT NORMS")
print("=" * 70)

print()
print(
    "Gradient norm = total L2 norm of gradients reaching "
    "the CNN parameters."
)

print()


gradient_norms = {}


# ------------------------------------------------------------
# Individual loss gradients
# ------------------------------------------------------------

for name in [
    "illumination",
    "exposure",
    "spatial",
    "color",
    "noise",
    "edge"
]:

    gradient_norm = compute_gradient_norm(
        losses[name],
        model
    )

    gradient_norms[name] = gradient_norm

    print(
        f"{name:<15}: "
        f"{gradient_norm:.10f}"
    )


# ------------------------------------------------------------
# Total loss gradient
# ------------------------------------------------------------

total_gradient_norm = compute_gradient_norm(
    losses["total"],
    model
)

gradient_norms["total"] = total_gradient_norm

print(
    f"{'total':<15}: "
    f"{total_gradient_norm:.10f}"
)


# ============================================================
# RELATIVE GRADIENT CONTRIBUTIONS
# ============================================================

individual_names = [
    "illumination",
    "exposure",
    "spatial",
    "color",
    "noise",
    "edge"
]

individual_gradient_sum = sum(
    gradient_norms[name]
    for name in individual_names
)


print()
print("=" * 70)
print("RELATIVE GRADIENT CONTRIBUTIONS")
print("=" * 70)

print()

if individual_gradient_sum > 0.0:

    for name in individual_names:

        contribution = (
            gradient_norms[name]
            / individual_gradient_sum
        ) * 100.0

        print(
            f"{name:<15}: "
            f"{contribution:8.3f}%"
        )

else:

    print(
        "All individual gradient norms are zero."
    )


# ============================================================
# CNN PARAMETER GRADIENT CHECK
# ============================================================

print()
print("=" * 70)
print("CNN PARAMETER GRADIENT CHECK")
print("=" * 70)

print()

model.zero_grad(
    set_to_none=True
)

losses["total"].backward()

for name, parameter in model.named_parameters():

    if parameter.grad is None:

        print(
            f"{name:<40}: "
            f"NO GRADIENT"
        )

    else:

        print(
            f"{name:<40}: "
            f"mean={parameter.grad.mean().item():+.10f}, "
            f"abs_mean={parameter.grad.abs().mean().item():.10f}, "
            f"max={parameter.grad.abs().max().item():.10f}"
        )


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 70)
print("GRADIENT DIAGNOSIS COMPLETED")
print("=" * 70)