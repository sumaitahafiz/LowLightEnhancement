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
from losses.color_loss import ColorLoss


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
# HELPER FUNCTIONS
# ============================================================

def get_channel_means(image):
    """
    Return mean value of R, G, and B channels.
    """

    means = image.mean(
        dim=(2, 3),
        keepdim=False
    )

    return means[0]


def get_channel_ratios(image):
    """
    Return normalized RGB channel ratios used by ColorLoss.
    """

    channel_means = image.mean(
        dim=(2, 3),
        keepdim=True
    )

    channel_sum = (
        channel_means.sum(
            dim=1,
            keepdim=True
        )
        + 1e-6
    )

    ratios = (
        channel_means
        / channel_sum
    )

    return ratios[0, :, 0, 0]


def print_channel_statistics(name, image):
    """
    Print channel means and normalized RGB ratios.
    """

    means = get_channel_means(image)
    ratios = get_channel_ratios(image)

    print()
    print(name)
    print("-" * 70)

    print(
        f"Mean R : {means[0].item():.8f}"
    )

    print(
        f"Mean G : {means[1].item():.8f}"
    )

    print(
        f"Mean B : {means[2].item():.8f}"
    )

    print()

    print(
        f"Ratio R: {ratios[0].item():.8f}"
    )

    print(
        f"Ratio G: {ratios[1].item():.8f}"
    )

    print(
        f"Ratio B: {ratios[2].item():.8f}"
    )

    print()

    print(
        f"R/G   : "
        f"{(means[0] / (means[1] + 1e-6)).item():.8f}"
    )

    print(
        f"B/G   : "
        f"{(means[2] / (means[1] + 1e-6)).item():.8f}"
    )

    print(
        f"B/R   : "
        f"{(means[2] / (means[0] + 1e-6)).item():.8f}"
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("COLOR LOSS DIAGNOSIS")
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
# ORIGINAL IMAGE STATISTICS
# ============================================================

print()
print("=" * 70)
print("ORIGINAL IMAGE COLOR STATISTICS")
print("=" * 70)

print_channel_statistics(
    "Original Low-Light Input",
    original
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

model.eval()


print()
print("Model loaded successfully.")


# ============================================================
# CREATE ENHANCEMENT + REFINEMENT
# ============================================================

enhancement = AdaptiveEnhancement().to(
    DEVICE
)

refinement = ImageRefinement().to(
    DEVICE
)

with torch.no_grad():

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
# ENHANCED IMAGE STATISTICS
# ============================================================

print()
print("=" * 70)
print("ENHANCED IMAGE COLOR STATISTICS")
print("=" * 70)

print_channel_statistics(
    "Enhanced Image",
    enhanced
)


# ============================================================
# REFINED IMAGE STATISTICS
# ============================================================

print()
print("=" * 70)
print("REFINED IMAGE COLOR STATISTICS")
print("=" * 70)

print_channel_statistics(
    "Refined Image",
    refined
)


# ============================================================
# COLOR LOSS
# ============================================================

color_loss = ColorLoss().to(
    DEVICE
)


# ============================================================
# COLOR LOSS VALUES
# ============================================================

original_vs_enhanced = color_loss(
    original,
    enhanced
)

original_vs_refined = color_loss(
    original,
    refined
)


print()
print("=" * 70)
print("COLOR LOSS VALUES")
print("=" * 70)

print()

print(
    f"Original vs Enhanced : "
    f"{original_vs_enhanced.item():.10f}"
)

print(
    f"Original vs Refined  : "
    f"{original_vs_refined.item():.10f}"
)


# ============================================================
# COLOR RATIO DIFFERENCES
# ============================================================

original_ratios = get_channel_ratios(
    original
)

enhanced_ratios = get_channel_ratios(
    enhanced
)

refined_ratios = get_channel_ratios(
    refined
)


print()
print("=" * 70)
print("RGB RATIO DIFFERENCES")
print("=" * 70)

print()

print(
    f"R difference "
    f"(Enhanced - Original): "
    f"{(enhanced_ratios[0] - original_ratios[0]).item():+.10f}"
)

print(
    f"G difference "
    f"(Enhanced - Original): "
    f"{(enhanced_ratios[1] - original_ratios[1]).item():+.10f}"
)

print(
    f"B difference "
    f"(Enhanced - Original): "
    f"{(enhanced_ratios[2] - original_ratios[2]).item():+.10f}"
)

print()

print(
    f"R difference "
    f"(Refined - Original): "
    f"{(refined_ratios[0] - original_ratios[0]).item():+.10f}"
)

print(
    f"G difference "
    f"(Refined - Original): "
    f"{(refined_ratios[1] - original_ratios[1]).item():+.10f}"
)

print(
    f"B difference "
    f"(Refined - Original): "
    f"{(refined_ratios[2] - original_ratios[2]).item():+.10f}"
)


# ============================================================
# COLOR LOSS GRADIENT ANALYSIS
# ============================================================

print()
print("=" * 70)
print("COLOR LOSS GRADIENT ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# Recreate forward pass with gradients
# ------------------------------------------------------------

model.train()

parameters = model(features)

enhanced_gradient = enhancement(
    rgb,
    illumination,
    parameters
)

refined_gradient = refinement(
    rgb,
    enhanced_gradient
)

color_loss_value = color_loss(
    rgb,
    refined_gradient
)


# ------------------------------------------------------------
# Gradient with respect to CNN parameters
# ------------------------------------------------------------

model.zero_grad(
    set_to_none=True
)

color_loss_value.backward()


print()
print(
    "CNN parameter gradients from ColorLoss:"
)

print()

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
# PARAMETER HEAD GRADIENTS
# ============================================================

print()
print("=" * 70)
print("PARAMETER HEAD GRADIENTS")
print("=" * 70)

print()

for name, parameter in model.named_parameters():

    if "parameter_head" in name:

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
print("COLOR DIAGNOSIS COMPLETED")
print("=" * 70)