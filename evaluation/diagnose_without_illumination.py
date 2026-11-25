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
# STORAGE
# ============================================================

normal_parameters = []
without_illumination_parameters = []

normal_strengths = []
without_illumination_strengths = []


# ============================================================
# PROCESS DATASET
# ============================================================

image_paths = sorted(
    LOL_V1_LOW_PATH.glob("*.png")
)

print("============================================================")
print("WITHOUT-ILLUMINATION CONTROLLED DIAGNOSTIC")
print("============================================================")
print(f"Number of images : {len(image_paths)}")
print(f"Checkpoint       : {CHECKPOINT_PATH}")
print(f"Device           : {DEVICE}")
print("============================================================")


with torch.no_grad():

    for image_path in image_paths:

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(image)

        # ----------------------------------------------------
        # Create normal six-channel feature tensor
        # ----------------------------------------------------

        features = create_analysis_features(
            image
        )

        features = features.unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # NORMAL CASE
        # ----------------------------------------------------

        normal_output = model(
            features
        )

        normal_strength = torch.sigmoid(
            normal_output
        )

        normal_parameter_mean = normal_output.mean(
            dim=(0, 2, 3)
        )

        normal_strength_mean = normal_strength.mean(
            dim=(0, 2, 3)
        )

        normal_parameters.append(
            normal_parameter_mean.cpu()
        )

        normal_strengths.append(
            normal_strength_mean.cpu()
        )

        # ----------------------------------------------------
        # WITHOUT ILLUMINATION
        #
        # Keep all other feature channels unchanged.
        # Only channel 3 is replaced by zeros.
        #
        # Channel mapping:
        # 0 = Red
        # 1 = Green
        # 2 = Blue
        # 3 = Illumination
        # 4 = High-frequency
        # 5 = Sobel edge
        # ----------------------------------------------------

        features_without_illumination = features.clone()

        features_without_illumination[
            :, 3:4, :, :
        ] = 0.0

        without_illumination_output = model(
            features_without_illumination
        )

        without_illumination_strength = torch.sigmoid(
            without_illumination_output
        )

        without_illumination_parameter_mean = (
            without_illumination_output.mean(
                dim=(0, 2, 3)
            )
        )

        without_illumination_strength_mean = (
            without_illumination_strength.mean(
                dim=(0, 2, 3)
            )
        )

        without_illumination_parameters.append(
            without_illumination_parameter_mean.cpu()
        )

        without_illumination_strengths.append(
            without_illumination_strength_mean.cpu()
        )


# ============================================================
# CONVERT TO TENSORS
# ============================================================

normal_parameters = torch.stack(
    normal_parameters
)

without_illumination_parameters = torch.stack(
    without_illumination_parameters
)

normal_strengths = torch.stack(
    normal_strengths
)

without_illumination_strengths = torch.stack(
    without_illumination_strengths
)


# ============================================================
# DATASET AVERAGES
# ============================================================

avg_normal_parameters = normal_parameters.mean(
    dim=0
)

avg_without_illumination_parameters = (
    without_illumination_parameters.mean(
        dim=0
    )
)

avg_normal_strengths = normal_strengths.mean(
    dim=0
)

avg_without_illumination_strengths = (
    without_illumination_strengths.mean(
        dim=0
    )
)


# ============================================================
# DIFFERENCES
# ============================================================

parameter_difference = (
    avg_without_illumination_parameters
    - avg_normal_parameters
)

strength_difference = (
    avg_without_illumination_strengths
    - avg_normal_strengths
)


# ============================================================
# PRINT NORMAL PARAMETERS
# ============================================================

print()
print("NORMAL CNN PARAMETERS")
print("------------------------------------------------------------")

print(
    f"R : {avg_normal_parameters[0].item():+.6f}"
)

print(
    f"G : {avg_normal_parameters[1].item():+.6f}"
)

print(
    f"B : {avg_normal_parameters[2].item():+.6f}"
)


# ============================================================
# PRINT WITHOUT ILLUMINATION PARAMETERS
# ============================================================

print()
print("CNN PARAMETERS WITHOUT ILLUMINATION")
print("------------------------------------------------------------")

print(
    f"R : {avg_without_illumination_parameters[0].item():+.6f}"
)

print(
    f"G : {avg_without_illumination_parameters[1].item():+.6f}"
)

print(
    f"B : {avg_without_illumination_parameters[2].item():+.6f}"
)


# ============================================================
# PARAMETER CHANGES
# ============================================================

print()
print("PARAMETER CHANGE AFTER REMOVING ILLUMINATION")
print("------------------------------------------------------------")

print(
    f"R change : {parameter_difference[0].item():+.6f}"
)

print(
    f"G change : {parameter_difference[1].item():+.6f}"
)

print(
    f"B change : {parameter_difference[2].item():+.6f}"
)


# ============================================================
# NORMAL STRENGTHS
# ============================================================

print()
print("NORMAL SIGMOID ENHANCEMENT STRENGTH")
print("------------------------------------------------------------")

print(
    f"R : {avg_normal_strengths[0].item():.6f}"
)

print(
    f"G : {avg_normal_strengths[1].item():.6f}"
)

print(
    f"B : {avg_normal_strengths[2].item():.6f}"
)


# ============================================================
# WITHOUT ILLUMINATION STRENGTHS
# ============================================================

print()
print("SIGMOID ENHANCEMENT STRENGTH WITHOUT ILLUMINATION")
print("------------------------------------------------------------")

print(
    f"R : {avg_without_illumination_strengths[0].item():.6f}"
)

print(
    f"G : {avg_without_illumination_strengths[1].item():.6f}"
)

print(
    f"B : {avg_without_illumination_strengths[2].item():.6f}"
)


# ============================================================
# STRENGTH CHANGES
# ============================================================

print()
print("ENHANCEMENT STRENGTH CHANGE AFTER REMOVING ILLUMINATION")
print("------------------------------------------------------------")

print(
    f"R change : {strength_difference[0].item():+.6f}"
)

print(
    f"G change : {strength_difference[1].item():+.6f}"
)

print(
    f"B change : {strength_difference[2].item():+.6f}"
)


# ============================================================
# BLUE BIAS COMPARISON
# ============================================================

normal_b_minus_r = (
    avg_normal_strengths[2]
    - avg_normal_strengths[0]
)

normal_b_minus_g = (
    avg_normal_strengths[2]
    - avg_normal_strengths[1]
)

without_b_minus_r = (
    avg_without_illumination_strengths[2]
    - avg_without_illumination_strengths[0]
)

without_b_minus_g = (
    avg_without_illumination_strengths[2]
    - avg_without_illumination_strengths[1]
)


print()
print("BLUE BIAS COMPARISON")
print("------------------------------------------------------------")

print(
    f"Normal B - R strength : "
    f"{normal_b_minus_r.item():+.6f}"
)

print(
    f"Without illumination B - R : "
    f"{without_b_minus_r.item():+.6f}"
)

print()

print(
    f"Normal B - G strength : "
    f"{normal_b_minus_g.item():+.6f}"
)

print(
    f"Without illumination B - G : "
    f"{without_b_minus_g.item():+.6f}"
)


# ============================================================
# BLUE BIAS REDUCTION
# ============================================================

blue_bias_reduction_br = (
    normal_b_minus_r
    - without_b_minus_r
)

blue_bias_reduction_bg = (
    normal_b_minus_g
    - without_b_minus_g
)


print()
print("BLUE BIAS REDUCTION")
print("------------------------------------------------------------")

print(
    f"B-R bias reduction : "
    f"{blue_bias_reduction_br.item():+.6f}"
)

print(
    f"B-G bias reduction : "
    f"{blue_bias_reduction_bg.item():+.6f}"
)


# ============================================================
# PER-IMAGE BLUE PARAMETER CHANGE
# ============================================================

blue_parameter_change = (
    without_illumination_parameters[:, 2]
    - normal_parameters[:, 2]
)

blue_strength_change = (
    without_illumination_strengths[:, 2]
    - normal_strengths[:, 2]
)


print()
print("PER-IMAGE BLUE PARAMETER CHANGE")
print("------------------------------------------------------------")

print(
    f"Minimum : {blue_parameter_change.min().item():+.6f}"
)

print(
    f"Maximum : {blue_parameter_change.max().item():+.6f}"
)

print(
    f"Mean    : {blue_parameter_change.mean().item():+.6f}"
)


print()
print("PER-IMAGE BLUE STRENGTH CHANGE")
print("------------------------------------------------------------")

print(
    f"Minimum : {blue_strength_change.min().item():+.6f}"
)

print(
    f"Maximum : {blue_strength_change.max().item():+.6f}"
)

print(
    f"Mean    : {blue_strength_change.mean().item():+.6f}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("CONTROLLED EXPERIMENT COMPLETE")

print(
    "Only the illumination feature was changed."
)

print(
    "All RGB, high-frequency, and Sobel features were kept unchanged."
)

print(
    "The trained model and checkpoint were NOT modified."
)
