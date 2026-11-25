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

LOL_LOW_PATH = (
    Path(r"D:\12th Trimester\DIP\DIP Paper\Datasets")
    / "LOLdataset"
    / "eval15"
    / "low"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 256
DEVICE = torch.device("cpu")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_image(image_path):
    """
    Load and resize one RGB image.
    """

    image = Image.open(
        image_path
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

    return image_tensor.unsqueeze(0).to(DEVICE)


def get_channel_means(image):
    """
    Return mean R, G, B values.
    """

    return image.mean(
        dim=(2, 3)
    )[0]


def get_channel_ratios(image):
    """
    Return normalized R, G, B ratios.
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


def get_color_loss(
    color_loss,
    original,
    image
):
    """
    Calculate the ColorLoss between the
    original image and another image.
    """

    return color_loss(
        original,
        image
    ).item()


def print_progress(
    index,
    total,
    image_name
):
    """
    Print dataset processing progress.
    """

    print(
        f"Processing "
        f"{index:2d}/{total}: "
        f"{image_name}"
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DATASET-WIDE COLOR DIAGNOSIS")
print("=" * 70)

print()
print("Checkpoint:")
print(CHECKPOINT_PATH)

print()
print("Dataset:")
print(LOL_LOW_PATH)

print()
print("Device:")
print(DEVICE)


# ============================================================
# FIND IMAGES
# ============================================================

image_paths = sorted(
    LOL_LOW_PATH.glob("*.png")
)

if len(image_paths) == 0:

    raise RuntimeError(
        f"No PNG images found in:\n"
        f"{LOL_LOW_PATH}"
    )


print()
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print()
print(
    f"Number of images: "
    f"{len(image_paths)}"
)


# ============================================================
# LOAD MODEL
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


# ============================================================
# PIPELINE MODULES
# ============================================================

enhancement = AdaptiveEnhancement().to(
    DEVICE
)

refinement = ImageRefinement().to(
    DEVICE
)

color_loss = ColorLoss().to(
    DEVICE
)


print()
print("Model loaded successfully.")


# ============================================================
# STORAGE
# ============================================================

original_means = []
enhanced_means = []
refined_means = []

original_ratios = []
enhanced_ratios = []
refined_ratios = []

enhanced_color_losses = []
refined_color_losses = []


# ============================================================
# PROCESS DATASET
# ============================================================

print()
print("=" * 70)
print("PROCESSING LOL-V1 EVAL15")
print("=" * 70)

with torch.no_grad():

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print_progress(
            index,
            len(image_paths),
            image_path.name
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        original = load_image(
            image_path
        )

        # ----------------------------------------------------
        # Remove batch dimension for feature generation
        # ----------------------------------------------------

        image_tensor = original[0]

        # ----------------------------------------------------
        # Create analysis features
        # ----------------------------------------------------

        features = create_analysis_features(
            image_tensor
        )

        features = (
            features
            .unsqueeze(0)
            .to(DEVICE)
        )

        rgb = features[
            :, 0:3, :, :
        ]

        illumination = features[
            :, 3:4, :, :
        ]

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        parameters = model(
            features
        )

        # ----------------------------------------------------
        # Adaptive enhancement
        # ----------------------------------------------------

        enhanced = enhancement(
            rgb,
            illumination,
            parameters
        )

        # ----------------------------------------------------
        # Refinement
        # ----------------------------------------------------

        refined = refinement(
            rgb,
            enhanced
        )

        # ----------------------------------------------------
        # Channel means
        # ----------------------------------------------------

        original_mean = (
            get_channel_means(rgb)
            .cpu()
        )

        enhanced_mean = (
            get_channel_means(enhanced)
            .cpu()
        )

        refined_mean = (
            get_channel_means(refined)
            .cpu()
        )

        # ----------------------------------------------------
        # Channel ratios
        # ----------------------------------------------------

        original_ratio = (
            get_channel_ratios(rgb)
            .cpu()
        )

        enhanced_ratio = (
            get_channel_ratios(enhanced)
            .cpu()
        )

        refined_ratio = (
            get_channel_ratios(refined)
            .cpu()
        )

        # ----------------------------------------------------
        # Color losses
        # ----------------------------------------------------

        enhanced_loss = get_color_loss(
            color_loss,
            rgb,
            enhanced
        )

        refined_loss = get_color_loss(
            color_loss,
            rgb,
            refined
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        original_means.append(
            original_mean
        )

        enhanced_means.append(
            enhanced_mean
        )

        refined_means.append(
            refined_mean
        )

        original_ratios.append(
            original_ratio
        )

        enhanced_ratios.append(
            enhanced_ratio
        )

        refined_ratios.append(
            refined_ratio
        )

        enhanced_color_losses.append(
            enhanced_loss
        )

        refined_color_losses.append(
            refined_loss
        )


# ============================================================
# CONVERT RESULTS TO TENSORS
# ============================================================

original_means = torch.stack(
    original_means
)

enhanced_means = torch.stack(
    enhanced_means
)

refined_means = torch.stack(
    refined_means
)

original_ratios = torch.stack(
    original_ratios
)

enhanced_ratios = torch.stack(
    enhanced_ratios
)

refined_ratios = torch.stack(
    refined_ratios
)


# ============================================================
# DATASET AVERAGES
# ============================================================

avg_original_means = (
    original_means.mean(dim=0)
)

avg_enhanced_means = (
    enhanced_means.mean(dim=0)
)

avg_refined_means = (
    refined_means.mean(dim=0)
)

avg_original_ratios = (
    original_ratios.mean(dim=0)
)

avg_enhanced_ratios = (
    enhanced_ratios.mean(dim=0)
)

avg_refined_ratios = (
    refined_ratios.mean(dim=0)
)

avg_enhanced_color_loss = (
    sum(enhanced_color_losses)
    / len(enhanced_color_losses)
)

avg_refined_color_loss = (
    sum(refined_color_losses)
    / len(refined_color_losses)
)


# ============================================================
# DATASET-WIDE CHANNEL MEANS
# ============================================================

print()
print("=" * 70)
print("AVERAGE CHANNEL MEANS")
print("=" * 70)

print()

print(
    f"Original : "
    f"R={avg_original_means[0].item():.8f}  "
    f"G={avg_original_means[1].item():.8f}  "
    f"B={avg_original_means[2].item():.8f}"
)

print(
    f"Enhanced : "
    f"R={avg_enhanced_means[0].item():.8f}  "
    f"G={avg_enhanced_means[1].item():.8f}  "
    f"B={avg_enhanced_means[2].item():.8f}"
)

print(
    f"Refined  : "
    f"R={avg_refined_means[0].item():.8f}  "
    f"G={avg_refined_means[1].item():.8f}  "
    f"B={avg_refined_means[2].item():.8f}"
)


# ============================================================
# DATASET-WIDE RGB RATIOS
# ============================================================

print()
print("=" * 70)
print("AVERAGE RGB RATIOS")
print("=" * 70)

print()

print(
    f"Original : "
    f"R={avg_original_ratios[0].item():.8f}  "
    f"G={avg_original_ratios[1].item():.8f}  "
    f"B={avg_original_ratios[2].item():.8f}"
)

print(
    f"Enhanced : "
    f"R={avg_enhanced_ratios[0].item():.8f}  "
    f"G={avg_enhanced_ratios[1].item():.8f}  "
    f"B={avg_enhanced_ratios[2].item():.8f}"
)

print(
    f"Refined  : "
    f"R={avg_refined_ratios[0].item():.8f}  "
    f"G={avg_refined_ratios[1].item():.8f}  "
    f"B={avg_refined_ratios[2].item():.8f}"
)


# ============================================================
# AVERAGE COLOR RATIOS
# ============================================================

original_b_over_g = (
    avg_original_means[2]
    / (avg_original_means[1] + 1e-6)
)

enhanced_b_over_g = (
    avg_enhanced_means[2]
    / (avg_enhanced_means[1] + 1e-6)
)

refined_b_over_g = (
    avg_refined_means[2]
    / (avg_refined_means[1] + 1e-6)
)

original_b_over_r = (
    avg_original_means[2]
    / (avg_original_means[0] + 1e-6)
)

enhanced_b_over_r = (
    avg_enhanced_means[2]
    / (avg_enhanced_means[0] + 1e-6)
)

refined_b_over_r = (
    avg_refined_means[2]
    / (avg_refined_means[0] + 1e-6)
)


print()
print("=" * 70)
print("AVERAGE BLUE CHANNEL RATIOS")
print("=" * 70)

print()

print(
    f"Original B/G : "
    f"{original_b_over_g.item():.8f}"
)

print(
    f"Enhanced B/G : "
    f"{enhanced_b_over_g.item():.8f}"
)

print(
    f"Refined B/G  : "
    f"{refined_b_over_g.item():.8f}"
)

print()

print(
    f"Original B/R : "
    f"{original_b_over_r.item():.8f}"
)

print(
    f"Enhanced B/R : "
    f"{enhanced_b_over_r.item():.8f}"
)

print(
    f"Refined B/R  : "
    f"{refined_b_over_r.item():.8f}"
)


# ============================================================
# COLOR LOSS RESULTS
# ============================================================

print()
print("=" * 70)
print("AVERAGE COLOR LOSS")
print("=" * 70)

print()

print(
    f"Original vs Enhanced : "
    f"{avg_enhanced_color_loss:.10f}"
)

print(
    f"Original vs Refined  : "
    f"{avg_refined_color_loss:.10f}"
)

print()

print(
    f"Refinement change    : "
    f"{avg_refined_color_loss - avg_enhanced_color_loss:+.10f}"
)


# ============================================================
# CHANNEL CHANGES
# ============================================================

print()
print("=" * 70)
print("AVERAGE CHANNEL CHANGES")
print("=" * 70)

print()

print(
    f"Enhanced R change : "
    f"{(avg_enhanced_means[0] - avg_original_means[0]).item():+.8f}"
)

print(
    f"Enhanced G change : "
    f"{(avg_enhanced_means[1] - avg_original_means[1]).item():+.8f}"
)

print(
    f"Enhanced B change : "
    f"{(avg_enhanced_means[2] - avg_original_means[2]).item():+.8f}"
)

print()

print(
    f"Refined R change  : "
    f"{(avg_refined_means[0] - avg_original_means[0]).item():+.8f}"
)

print(
    f"Refined G change  : "
    f"{(avg_refined_means[1] - avg_original_means[1]).item():+.8f}"
)

print(
    f"Refined B change  : "
    f"{(avg_refined_means[2] - avg_original_means[2]).item():+.8f}"
)


# ============================================================
# PER-IMAGE EXTREME CHECK
# ============================================================

enhanced_b_over_g_all = (
    enhanced_means[:, 2]
    / (enhanced_means[:, 1] + 1e-6)
)

refined_b_over_g_all = (
    refined_means[:, 2]
    / (refined_means[:, 1] + 1e-6)
)

original_b_over_g_all = (
    original_means[:, 2]
    / (original_means[:, 1] + 1e-6)
)


print()
print("=" * 70)
print("PER-IMAGE B/G RANGE")
print("=" * 70)

print()

print(
    f"Original B/G : "
    f"min={original_b_over_g_all.min().item():.8f}, "
    f"max={original_b_over_g_all.max().item():.8f}"
)

print(
    f"Enhanced B/G : "
    f"min={enhanced_b_over_g_all.min().item():.8f}, "
    f"max={enhanced_b_over_g_all.max().item():.8f}"
)

print(
    f"Refined B/G  : "
    f"min={refined_b_over_g_all.min().item():.8f}, "
    f"max={refined_b_over_g_all.max().item():.8f}"
)


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 70)
print("DATASET-WIDE COLOR DIAGNOSIS COMPLETED")
print("=" * 70)