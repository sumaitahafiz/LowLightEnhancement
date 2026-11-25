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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# ============================================================
# LOAD MODEL
# ============================================================

model = LightweightZeroReferenceCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ============================================================
# STORAGE
# ============================================================

parameter_means = []
parameter_stds = []

strength_means = []
strength_stds = []

blue_minus_red_strength = []
blue_minus_green_strength = []

blue_greater_red_percent = []
blue_greater_green_percent = []

blue_above_075_percent = []
blue_above_090_percent = []


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

image_paths = sorted(LOL_V1_LOW_PATH.glob("*.png"))

print("============================================================")
print("CNN PARAMETER DATASET DIAGNOSTIC")
print("============================================================")
print(f"Number of images : {len(image_paths)}")
print(f"Checkpoint       : {CHECKPOINT_PATH}")
print(f"Device           : {DEVICE}")
print("============================================================")


with torch.no_grad():

    for image_path in image_paths:

        image = Image.open(image_path).convert("RGB")
        image = transform(image)

        # ----------------------------------------------------
        # Create the same six-channel feature representation
        # used during training.
        # ----------------------------------------------------

        features = create_analysis_features(image)

        features = features.unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # CNN enhancement parameters
        # ----------------------------------------------------

        parameters = model(features)

        # ----------------------------------------------------
        # Convert raw parameters to enhancement strengths
        # ----------------------------------------------------

        strength = torch.sigmoid(parameters)

        # ----------------------------------------------------
        # Raw parameter statistics
        # ----------------------------------------------------

        parameter_mean = parameters.mean(
            dim=(0, 2, 3)
        )

        parameter_std = parameters.std(
            dim=(0, 2, 3)
        )

        parameter_means.append(
            parameter_mean.cpu()
        )

        parameter_stds.append(
            parameter_std.cpu()
        )

        # ----------------------------------------------------
        # Enhancement strength statistics
        # ----------------------------------------------------

        strength_mean = strength.mean(
            dim=(0, 2, 3)
        )

        strength_std = strength.std(
            dim=(0, 2, 3)
        )

        strength_means.append(
            strength_mean.cpu()
        )

        strength_stds.append(
            strength_std.cpu()
        )

        # ----------------------------------------------------
        # Channel differences
        # ----------------------------------------------------

        red_strength = strength[:, 0:1, :, :]
        green_strength = strength[:, 1:2, :, :]
        blue_strength = strength[:, 2:3, :, :]

        b_minus_r = (
            blue_strength - red_strength
        ).mean()

        b_minus_g = (
            blue_strength - green_strength
        ).mean()

        blue_minus_red_strength.append(
            b_minus_r.item()
        )

        blue_minus_green_strength.append(
            b_minus_g.item()
        )

        # ----------------------------------------------------
        # Percentage of pixels where blue enhancement is
        # stronger than red and green.
        # ----------------------------------------------------

        blue_greater_red = (
            blue_strength > red_strength
        ).float().mean() * 100.0

        blue_greater_green = (
            blue_strength > green_strength
        ).float().mean() * 100.0

        blue_greater_red_percent.append(
            blue_greater_red.item()
        )

        blue_greater_green_percent.append(
            blue_greater_green.item()
        )

        # ----------------------------------------------------
        # Percentage of pixels with strong blue enhancement.
        # ----------------------------------------------------

        blue_above_075 = (
            blue_strength > 0.75
        ).float().mean() * 100.0

        blue_above_090 = (
            blue_strength > 0.90
        ).float().mean() * 100.0

        blue_above_075_percent.append(
            blue_above_075.item()
        )

        blue_above_090_percent.append(
            blue_above_090.item()
        )


# ============================================================
# CONVERT DATA TO TENSORS
# ============================================================

parameter_means = torch.stack(parameter_means)
parameter_stds = torch.stack(parameter_stds)

strength_means = torch.stack(strength_means)
strength_stds = torch.stack(strength_stds)

blue_minus_red_strength = torch.tensor(
    blue_minus_red_strength
)

blue_minus_green_strength = torch.tensor(
    blue_minus_green_strength
)

blue_greater_red_percent = torch.tensor(
    blue_greater_red_percent
)

blue_greater_green_percent = torch.tensor(
    blue_greater_green_percent
)

blue_above_075_percent = torch.tensor(
    blue_above_075_percent
)

blue_above_090_percent = torch.tensor(
    blue_above_090_percent
)


# ============================================================
# DATASET AVERAGES
# ============================================================

avg_parameter_mean = parameter_means.mean(dim=0)
avg_parameter_std = parameter_stds.mean(dim=0)

avg_strength_mean = strength_means.mean(dim=0)
avg_strength_std = strength_stds.mean(dim=0)

avg_b_minus_r = blue_minus_red_strength.mean()
avg_b_minus_g = blue_minus_green_strength.mean()

avg_blue_greater_red = blue_greater_red_percent.mean()
avg_blue_greater_green = blue_greater_green_percent.mean()

avg_blue_above_075 = blue_above_075_percent.mean()
avg_blue_above_090 = blue_above_090_percent.mean()


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("RAW CNN PARAMETER MEANS")
print("--------------------------------------------")
print(f"R : {avg_parameter_mean[0].item():.6f}")
print(f"G : {avg_parameter_mean[1].item():.6f}")
print(f"B : {avg_parameter_mean[2].item():.6f}")

print()
print("RAW CNN PARAMETER STANDARD DEVIATIONS")
print("--------------------------------------------")
print(f"R : {avg_parameter_std[0].item():.6f}")
print(f"G : {avg_parameter_std[1].item():.6f}")
print(f"B : {avg_parameter_std[2].item():.6f}")


print()
print("SIGMOID ENHANCEMENT STRENGTH MEANS")
print("--------------------------------------------")
print(f"R : {avg_strength_mean[0].item():.6f}")
print(f"G : {avg_strength_mean[1].item():.6f}")
print(f"B : {avg_strength_mean[2].item():.6f}")


print()
print("SIGMOID ENHANCEMENT STRENGTH STANDARD DEVIATIONS")
print("--------------------------------------------")
print(f"R : {avg_strength_std[0].item():.6f}")
print(f"G : {avg_strength_std[1].item():.6f}")
print(f"B : {avg_strength_std[2].item():.6f}")


print()
print("BLUE VS OTHER CHANNELS")
print("--------------------------------------------")
print(
    f"Average B - R strength : "
    f"{avg_b_minus_r.item():+.6f}"
)
print(
    f"Average B - G strength : "
    f"{avg_b_minus_g.item():+.6f}"
)


print()
print("PERCENTAGE OF PIXELS WHERE BLUE STRENGTH IS GREATER")
print("--------------------------------------------")
print(
    f"B > R : "
    f"{avg_blue_greater_red.item():.2f}%"
)
print(
    f"B > G : "
    f"{avg_blue_greater_green.item():.2f}%"
)


print()
print("STRONG BLUE ENHANCEMENT")
print("--------------------------------------------")
print(
    f"B strength > 0.75 : "
    f"{avg_blue_above_075.item():.2f}%"
)
print(
    f"B strength > 0.90 : "
    f"{avg_blue_above_090.item():.2f}%"
)


# ============================================================
# RANGE ACROSS THE DATASET
# ============================================================

print()
print("DATASET RANGE OF CHANNEL MEANS")
print("--------------------------------------------")

for channel, name in enumerate(["R", "G", "B"]):

    minimum = parameter_means[:, channel].min()
    maximum = parameter_means[:, channel].max()

    print(
        f"Parameter {name}: "
        f"min={minimum.item():.6f}, "
        f"max={maximum.item():.6f}"
    )


print()
print("DATASET RANGE OF SIGMOID STRENGTH")
print("--------------------------------------------")

for channel, name in enumerate(["R", "G", "B"]):

    minimum = strength_means[:, channel].min()
    maximum = strength_means[:, channel].max()

    print(
        f"Strength {name}: "
        f"min={minimum.item():.6f}, "
        f"max={maximum.item():.6f}"
    )


# ============================================================
# PER-IMAGE CONSISTENCY
# ============================================================

print()
print("PER-IMAGE BLUE BIAS CONSISTENCY")
print("--------------------------------------------")

print(
    f"B > R in at least 50% of pixels: "
    f"{(blue_greater_red_percent >= 50).sum().item()}"
    f"/{len(image_paths)} images"
)

print(
    f"B > G in at least 50% of pixels: "
    f"{(blue_greater_green_percent >= 50).sum().item()}"
    f"/{len(image_paths)} images"
)

print(
    f"B strength > 0.75 in at least 50% of pixels: "
    f"{(blue_above_075_percent >= 50).sum().item()}"
    f"/{len(image_paths)} images"
)


print()
print("============================================================")
print("DIAGNOSTIC COMPLETE")
print("============================================================")