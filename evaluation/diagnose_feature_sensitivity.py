import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torchvision import transforms

from model import LightweightZeroReferenceCNN
from preprocessing.fusion import create_analysis_features


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "epoch_5.pth"

LOL_V1_LOW_PATH = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\LOLdataset\eval15\low"
)

IMAGE_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = LightweightZeroReferenceCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = [
    "Red",
    "Green",
    "Blue",
    "Illumination",
    "High-frequency",
    "Sobel edge"
]


# ============================================================
# STORAGE
# ============================================================

# Gradient of average BLUE CNN parameter with respect to
# each input feature.
blue_gradient_means = []

# Absolute gradient magnitude for each feature.
blue_gradient_abs_means = []

# Gradient of the complete RGB output with respect to
# each input feature.
rgb_gradient_abs_means = []


# ============================================================
# PROCESS DATASET
# ============================================================

image_paths = sorted(
    LOL_V1_LOW_PATH.glob("*.png")
)

print("============================================================")
print("FEATURE SENSITIVITY DIAGNOSTIC")
print("============================================================")
print(f"Number of images : {len(image_paths)}")
print(f"Checkpoint       : {CHECKPOINT_PATH}")
print(f"Device           : {DEVICE}")
print("============================================================")


for image_path in image_paths:

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    # --------------------------------------------------------
    # Create the six-channel feature representation
    # --------------------------------------------------------

    features = create_analysis_features(
        image
    )

    features = features.unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # Enable gradient calculation for the input features
    # --------------------------------------------------------

    features.requires_grad_(True)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    parameters = model(features)

    # ========================================================
    # TEST 1:
    # Sensitivity of BLUE CNN parameter
    # ========================================================

    blue_output = parameters[:, 2:3, :, :].mean()

    model.zero_grad(set_to_none=True)

    if features.grad is not None:
        features.grad.zero_()

    blue_output.backward(
        retain_graph=True
    )

    blue_gradient = features.grad.detach()

    # Mean signed gradient for each feature.
    blue_gradient_mean = blue_gradient.mean(
        dim=(0, 2, 3)
    )

    # Mean absolute gradient for each feature.
    blue_gradient_abs_mean = blue_gradient.abs().mean(
        dim=(0, 2, 3)
    )

    blue_gradient_means.append(
        blue_gradient_mean.cpu()
    )

    blue_gradient_abs_means.append(
        blue_gradient_abs_mean.cpu()
    )

    # ========================================================
    # TEST 2:
    # Sensitivity of complete RGB CNN output
    # ========================================================

    features.grad.zero_()
    model.zero_grad(set_to_none=True)

    rgb_output = parameters.mean()

    rgb_output.backward()

    rgb_gradient = features.grad.detach()

    rgb_gradient_abs_mean = rgb_gradient.abs().mean(
        dim=(0, 2, 3)
    )

    rgb_gradient_abs_means.append(
        rgb_gradient_abs_mean.cpu()
    )


# ============================================================
# CONVERT TO TENSORS
# ============================================================

blue_gradient_means = torch.stack(
    blue_gradient_means
)

blue_gradient_abs_means = torch.stack(
    blue_gradient_abs_means
)

rgb_gradient_abs_means = torch.stack(
    rgb_gradient_abs_means
)


# ============================================================
# DATASET AVERAGES
# ============================================================

average_blue_gradient = blue_gradient_means.mean(
    dim=0
)

average_blue_gradient_abs = blue_gradient_abs_means.mean(
    dim=0
)

average_rgb_gradient_abs = rgb_gradient_abs_means.mean(
    dim=0
)


# ============================================================
# PRINT SIGNED BLUE GRADIENT
# ============================================================

print()
print("AVERAGE SIGNED GRADIENT OF BLUE CNN PARAMETER")
print("------------------------------------------------------------")

for index, name in enumerate(feature_names):

    print(
        f"{index}: {name:<18} "
        f"{average_blue_gradient[index].item():+.8f}"
    )


# ============================================================
# PRINT ABSOLUTE BLUE SENSITIVITY
# ============================================================

print()
print("AVERAGE ABSOLUTE BLUE PARAMETER SENSITIVITY")
print("------------------------------------------------------------")

for index, name in enumerate(feature_names):

    print(
        f"{index}: {name:<18} "
        f"{average_blue_gradient_abs[index].item():.8f}"
    )


# ============================================================
# NORMALIZED BLUE SENSITIVITY
# ============================================================

blue_total_sensitivity = (
    average_blue_gradient_abs.sum()
    + 1e-12
)

blue_relative_sensitivity = (
    average_blue_gradient_abs
    / blue_total_sensitivity
) * 100.0


print()
print("RELATIVE BLUE PARAMETER SENSITIVITY")
print("------------------------------------------------------------")

for index, name in enumerate(feature_names):

    print(
        f"{index}: {name:<18} "
        f"{blue_relative_sensitivity[index].item():.2f}%"
    )


# ============================================================
# COMPLETE RGB OUTPUT SENSITIVITY
# ============================================================

print()
print("AVERAGE ABSOLUTE SENSITIVITY OF COMPLETE RGB OUTPUT")
print("------------------------------------------------------------")

for index, name in enumerate(feature_names):

    print(
        f"{index}: {name:<18} "
        f"{average_rgb_gradient_abs[index].item():.8f}"
    )


# ============================================================
# NORMALIZED COMPLETE RGB SENSITIVITY
# ============================================================

rgb_total_sensitivity = (
    average_rgb_gradient_abs.sum()
    + 1e-12
)

rgb_relative_sensitivity = (
    average_rgb_gradient_abs
    / rgb_total_sensitivity
) * 100.0


print()
print("RELATIVE COMPLETE RGB OUTPUT SENSITIVITY")
print("------------------------------------------------------------")

for index, name in enumerate(feature_names):

    print(
        f"{index}: {name:<18} "
        f"{rgb_relative_sensitivity[index].item():.2f}%"
    )


# ============================================================
# MOST SENSITIVE FEATURES
# ============================================================

blue_order = torch.argsort(
    average_blue_gradient_abs,
    descending=True
)

rgb_order = torch.argsort(
    average_rgb_gradient_abs,
    descending=True
)


print()
print("FEATURES RANKED BY BLUE OUTPUT SENSITIVITY")
print("------------------------------------------------------------")

for rank, index in enumerate(blue_order):

    print(
        f"{rank + 1}. "
        f"{feature_names[index]} : "
        f"{average_blue_gradient_abs[index].item():.8f} "
        f"({blue_relative_sensitivity[index].item():.2f}%)"
    )


print()
print("FEATURES RANKED BY COMPLETE RGB OUTPUT SENSITIVITY")
print("------------------------------------------------------------")

for rank, index in enumerate(rgb_order):

    print(
        f"{rank + 1}. "
        f"{feature_names[index]} : "
        f"{average_rgb_gradient_abs[index].item():.8f} "
        f"({rgb_relative_sensitivity[index].item():.2f}%)"
    )


# ============================================================
# PER-IMAGE CONSISTENCY
# ============================================================

print()
print("PER-IMAGE BLUE SENSITIVITY CONSISTENCY")
print("------------------------------------------------------------")

for feature_index, feature_name in enumerate(feature_names):

    values = blue_gradient_abs_means[
        :, feature_index
    ]

    print(
        f"{feature_name:<18} "
        f"min={values.min().item():.8f}, "
        f"max={values.max().item():.8f}, "
        f"mean={values.mean().item():.8f}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

most_sensitive_index = torch.argmax(
    average_blue_gradient_abs
)

least_sensitive_index = torch.argmin(
    average_blue_gradient_abs
)

print()
print("============================================================")
print("DIAGNOSTIC SUMMARY")
print("============================================================")

print(
    "Most sensitive feature for BLUE CNN output: "
    f"{feature_names[most_sensitive_index]}"
)

print(
    "Least sensitive feature for BLUE CNN output: "
    f"{feature_names[least_sensitive_index]}"
)

print()
print(
    "This diagnostic measures input-feature sensitivity "
    "using gradients."
)

print(
    "It does NOT modify the model or checkpoint."
)

print("============================================================")
print("DIAGNOSTIC COMPLETE")
print("============================================================")