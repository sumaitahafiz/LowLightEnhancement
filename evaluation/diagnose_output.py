import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

# ============================================================
# PROJECT PATH
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


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "epoch_5.pth"
)

LOL_V1_LOW = (
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

def print_stats(name, tensor):
    tensor = tensor.detach().cpu()

    print(f"\n{name}")
    print("-" * 60)
    print(f"Shape : {tuple(tensor.shape)}")
    print(f"Min   : {tensor.min().item():.6f}")
    print(f"Max   : {tensor.max().item():.6f}")
    print(f"Mean  : {tensor.mean().item():.6f}")
    print(f"Std   : {tensor.std().item():.6f}")

    if tensor.ndim == 4 and tensor.shape[1] == 3:
        for i, channel in enumerate(["R", "G", "B"]):
            channel_data = tensor[:, i:i+1, :, :]
            print(
                f"{channel} mean : "
                f"{channel_data.mean().item():.6f}"
            )
            print(
                f"{channel} std  : "
                f"{channel_data.std().item():.6f}"
            )


# ============================================================
# LOAD IMAGE
# ============================================================

print("=" * 70)
print("ENHANCEMENT OUTPUT DIAGNOSIS")
print("=" * 70)

print(f"\nCheckpoint:")
print(CHECKPOINT_PATH)

print(f"\nTest image:")
print(LOL_V1_LOW)

print(f"\nDevice:")
print(DEVICE)


image = Image.open(LOL_V1_LOW).convert("RGB")
image = image.resize(
    (IMAGE_SIZE, IMAGE_SIZE),
    Image.Resampling.BILINEAR
)

image_np = np.asarray(image).astype(np.float32) / 255.0

rgb = torch.from_numpy(
    image_np.transpose(2, 0, 1)
).float()

rgb_batch = rgb.unsqueeze(0).to(DEVICE)

print_stats("Original Low-Light Input", rgb_batch)


# ============================================================
# ANALYSIS FEATURES
# ============================================================

features = create_analysis_features(rgb)

print_stats("Analysis Feature Tensor", features)

# Feature layout:
# 0-2 = RGB
# 3   = illumination
# 4   = high-frequency variation
# 5   = Sobel edge/detail

illumination = features[3:4].unsqueeze(0).to(DEVICE)

print_stats("Illumination Map", illumination)


# ============================================================
# LOAD MODEL
# ============================================================

model = LightweightZeroReferenceCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("\nModel loaded successfully.")


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    feature_batch = features.unsqueeze(0).to(DEVICE)

    parameters = model(feature_batch)

    print_stats(
        "CNN Enhancement Parameters",
        parameters
    )

    # --------------------------------------------------------
    # Sigmoid strength used by AdaptiveEnhancement
    # --------------------------------------------------------

    strength = torch.sigmoid(parameters)

    print_stats(
        "Sigmoid Enhancement Strength",
        strength
    )

    # --------------------------------------------------------
    # Adaptive enhancement
    # --------------------------------------------------------

    enhancement = AdaptiveEnhancement()

    enhanced = enhancement(
        rgb_batch,
        illumination,
        parameters
    )

    print_stats(
        "Enhanced Image",
        enhanced
    )

    # --------------------------------------------------------
    # Refinement
    # --------------------------------------------------------

    refinement = ImageRefinement()

    refined = refinement(
        rgb_batch,
        enhanced
    )

    print_stats(
        "Refined Image",
        refined
    )


# ============================================================
# CHANNEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("CHANNEL MEAN COMPARISON")
print("=" * 70)

for name, tensor in [
    ("Original", rgb_batch),
    ("Enhanced", enhanced),
    ("Refined", refined),
]:

    tensor = tensor.detach().cpu()

    means = tensor.mean(
        dim=(0, 2, 3)
    )

    print(
        f"{name:<10} "
        f"R={means[0].item():.6f}   "
        f"G={means[1].item():.6f}   "
        f"B={means[2].item():.6f}"
    )


# ============================================================
# CHANNEL DIFFERENCES
# ============================================================

print("\n" + "=" * 70)
print("CHANNEL DIFFERENCES")
print("=" * 70)

original_means = rgb_batch.detach().cpu().mean(
    dim=(0, 2, 3)
)

enhanced_means = enhanced.detach().cpu().mean(
    dim=(0, 2, 3)
)

refined_means = refined.detach().cpu().mean(
    dim=(0, 2, 3)
)

for i, channel in enumerate(["R", "G", "B"]):

    enhancement_change = (
        enhanced_means[i] - original_means[i]
    )

    refinement_change = (
        refined_means[i] - enhanced_means[i]
    )

    print(
        f"{channel}: "
        f"enhancement change = "
        f"{enhancement_change.item():+.6f}, "
        f"refinement change = "
        f"{refinement_change.item():+.6f}"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETED")
print("=" * 70)