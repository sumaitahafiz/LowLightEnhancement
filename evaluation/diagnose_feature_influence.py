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

feature_means = []
feature_stds = []

blue_parameter_means = []
red_parameter_means = []
green_parameter_means = []

blue_strength_means = []
red_strength_means = []
green_strength_means = []


# ============================================================
# PROCESS DATASET
# ============================================================

image_paths = sorted(
    LOL_V1_LOW_PATH.glob("*.png")
)

print("============================================================")
print("FEATURE / CNN OUTPUT DIAGNOSTIC")
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
    # Create six-channel feature tensor
    # --------------------------------------------------------

    features = create_analysis_features(
        image
    )

    features = features.unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # We need gradients for this diagnostic.
    #
    # The model is still evaluated in eval mode so that its
    # behavior matches the trained checkpoint.
    # --------------------------------------------------------

    features_for_gradient = features.clone()
    features_for_gradient.requires_grad_(True)

    parameters = model(
        features_for_gradient
    )

    # --------------------------------------------------------
    # Average CNN output for each RGB parameter channel
    # --------------------------------------------------------

    parameter_mean = parameters.mean(
        dim=(0, 2, 3)
    )

    red_parameter = parameter_mean[0]
    green_parameter = parameter_mean[1]
    blue_parameter = parameter_mean[2]

    red_parameter_means.append(
        red_parameter.detach().cpu()
    )

    green_parameter_means.append(
        green_parameter.detach().cpu()
    )

    blue_parameter_means.append(
        blue_parameter.detach().cpu()
    )

    # --------------------------------------------------------
    # Sigmoid enhancement strengths
    # --------------------------------------------------------

    strength = torch.sigmoid(
        parameters
    )

    strength_mean = strength.mean(
        dim=(0, 2, 3)
    )

    red_strength_means.append(
        strength_mean[0].detach().cpu()
    )

    green_strength_means.append(
        strength_mean[1].detach().cpu()
    )

    blue_strength_means.append(
        strength_mean[2].detach().cpu()
    )

    # --------------------------------------------------------
    # Feature statistics
    # --------------------------------------------------------

    current_feature_mean = features.mean(
        dim=(0, 2, 3)
    )

    current_feature_std = features.std(
        dim=(0, 2, 3)
    )

    feature_means.append(
        current_feature_mean.detach().cpu()
    )

    feature_stds.append(
        current_feature_std.detach().cpu()
    )


# ============================================================
# CONVERT TO TENSORS
# ============================================================

feature_means = torch.stack(
    feature_means
)

feature_stds = torch.stack(
    feature_stds
)

red_parameter_means = torch.stack(
    red_parameter_means
)

green_parameter_means = torch.stack(
    green_parameter_means
)

blue_parameter_means = torch.stack(
    blue_parameter_means
)

red_strength_means = torch.stack(
    red_strength_means
)

green_strength_means = torch.stack(
    green_strength_means
)

blue_strength_means = torch.stack(
    blue_strength_means
)


# ============================================================
# DATASET AVERAGES
# ============================================================

average_feature_means = feature_means.mean(
    dim=0
)

average_feature_stds = feature_stds.mean(
    dim=0
)

average_red_parameter = red_parameter_means.mean()
average_green_parameter = green_parameter_means.mean()
average_blue_parameter = blue_parameter_means.mean()

average_red_strength = red_strength_means.mean()
average_green_strength = green_strength_means.mean()
average_blue_strength = blue_strength_means.mean()


# ============================================================
# PRINT FEATURE STATISTICS
# ============================================================

print()
print("INPUT FEATURE STATISTICS")
print("------------------------------------------------------------")

for index, name in enumerate(feature_names):

    print(
        f"{index}: {name:<18} "
        f"mean={average_feature_means[index].item():.6f} "
        f"std={average_feature_stds[index].item():.6f}"
    )


# ============================================================
# PRINT CNN PARAMETER OUTPUT
# ============================================================

print()
print("AVERAGE CNN PARAMETER OUTPUT")
print("------------------------------------------------------------")

print(
    f"R : {average_red_parameter.item():+.6f}"
)

print(
    f"G : {average_green_parameter.item():+.6f}"
)

print(
    f"B : {average_blue_parameter.item():+.6f}"
)


# ============================================================
# PRINT ENHANCEMENT STRENGTH
# ============================================================

print()
print("AVERAGE SIGMOID ENHANCEMENT STRENGTH")
print("------------------------------------------------------------")

print(
    f"R : {average_red_strength.item():.6f}"
)

print(
    f"G : {average_green_strength.item():.6f}"
)

print(
    f"B : {average_blue_strength.item():.6f}"
)


# ============================================================
# BLUE BIAS
# ============================================================

print()
print("BLUE BIAS")
print("------------------------------------------------------------")

print(
    f"B - R parameter : "
    f"{(average_blue_parameter - average_red_parameter).item():+.6f}"
)

print(
    f"B - G parameter : "
    f"{(average_blue_parameter - average_green_parameter).item():+.6f}"
)

print(
    f"B - R strength : "
    f"{(average_blue_strength - average_red_strength).item():+.6f}"
)

print(
    f"B - G strength : "
    f"{(average_blue_strength - average_green_strength).item():+.6f}"
)


# ============================================================
# PER-IMAGE RANGE
# ============================================================

print()
print("PER-IMAGE PARAMETER RANGE")
print("------------------------------------------------------------")

print(
    f"R parameter: "
    f"min={red_parameter_means.min().item():+.6f}, "
    f"max={red_parameter_means.max().item():+.6f}"
)

print(
    f"G parameter: "
    f"min={green_parameter_means.min().item():+.6f}, "
    f"max={green_parameter_means.max().item():+.6f}"
)

print(
    f"B parameter: "
    f"min={blue_parameter_means.min().item():+.6f}, "
    f"max={blue_parameter_means.max().item():+.6f}"
)


print()
print("PER-IMAGE STRENGTH RANGE")
print("------------------------------------------------------------")

print(
    f"R strength: "
    f"min={red_strength_means.min().item():.6f}, "
    f"max={red_strength_means.max().item():.6f}"
)

print(
    f"G strength: "
    f"min={green_strength_means.min().item():.6f}, "
    f"max={green_strength_means.max().item():.6f}"
)

print(
    f"B strength: "
    f"min={blue_strength_means.min().item():.6f}, "
    f"max={blue_strength_means.max().item():.6f}"
)


# ============================================================
# FINAL INTERPRETATION VALUES
# ============================================================

print()
print("============================================================")
print("DIAGNOSTIC SUMMARY")
print("============================================================")

if average_blue_parameter > average_red_parameter:
    print(
        "Blue CNN parameter is higher than Red."
    )
else:
    print(
        "Blue CNN parameter is NOT higher than Red."
    )

if average_blue_parameter > average_green_parameter:
    print(
        "Blue CNN parameter is higher than Green."
    )
else:
    print(
        "Blue CNN parameter is NOT higher than Green."
    )

if average_blue_strength > average_red_strength:
    print(
        "Blue enhancement strength is higher than Red."
    )
else:
    print(
        "Blue enhancement strength is NOT higher than Red."
    )

if average_blue_strength > average_green_strength:
    print(
        "Blue enhancement strength is higher than Green."
    )
else:
    print(
        "Blue enhancement strength is NOT higher than Green."
    )

print()
print("============================================================")
print("DIAGNOSTIC COMPLETE")
print("============================================================")