import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from model import LightweightZeroReferenceCNN


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "epoch_5.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
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
# GET PARAMETER-HEAD WEIGHTS
# ============================================================

parameter_head = model.parameter_head[0]

weights = parameter_head.weight.detach().cpu()
bias = parameter_head.bias.detach().cpu()


# ============================================================
# BASIC INFORMATION
# ============================================================

print()
print("=" * 70)
print("PARAMETER HEAD WEIGHT DIAGNOSTIC")
print("=" * 70)

print()
print("Checkpoint:")
print(CHECKPOINT_PATH)

print()
print("Parameter-head weight shape:")
print(weights.shape)

print()
print("Parameter-head bias shape:")
print(bias.shape)


# ============================================================
# OUTPUT CHANNEL BIASES
# ============================================================

print()
print("=" * 70)
print("OUTPUT CHANNEL BIASES")
print("=" * 70)

print()
print(f"R bias : {bias[0].item():+.8f}")
print(f"G bias : {bias[1].item():+.8f}")
print(f"B bias : {bias[2].item():+.8f}")


# ============================================================
# WEIGHT STATISTICS FOR EACH OUTPUT CHANNEL
# ============================================================

print()
print("=" * 70)
print("WEIGHT STATISTICS BY OUTPUT CHANNEL")
print("=" * 70)

for channel, name in enumerate(["R", "G", "B"]):

    channel_weights = weights[channel]

    print()
    print(f"{name} OUTPUT")

    print(
        f"Mean       : {channel_weights.mean().item():+.8f}"
    )

    print(
        f"Std        : {channel_weights.std().item():.8f}"
    )

    print(
        f"Min        : {channel_weights.min().item():+.8f}"
    )

    print(
        f"Max        : {channel_weights.max().item():+.8f}"
    )

    print(
        f"Abs mean   : {channel_weights.abs().mean().item():.8f}"
    )

    print(
        f"L1 norm    : {channel_weights.abs().sum().item():.8f}"
    )


# ============================================================
# INPUT FEATURE CONTRIBUTION
# ============================================================

print()
print("=" * 70)
print("INPUT FEATURE CONTRIBUTION TO EACH OUTPUT")
print("=" * 70)

feature_names = [
    "Feature 0 - Red",
    "Feature 1 - Green",
    "Feature 2 - Blue",
    "Feature 3 - Illumination",
    "Feature 4 - High-frequency",
    "Feature 5 - Sobel edge"
]

# Each output has:
# [16 input feature maps, 3, 3]
#
# Calculate the average absolute weight for each
# input feature channel.

for output_channel, output_name in enumerate(
    ["R parameter", "G parameter", "B parameter"]
):

    print()
    print(output_name)

    output_weights = weights[output_channel]

    feature_strength = (
        output_weights
        .abs()
        .mean(dim=(1, 2))
    )

    total_strength = feature_strength.sum()

    for feature_index, feature_name in enumerate(
        feature_names
    ):

        value = feature_strength[feature_index].item()

        percentage = (
            100.0 * value / (total_strength.item() + 1e-12)
        )

        print(
            f"{feature_name:<30}"
            f" {value:.8f}"
            f" ({percentage:.2f}%)"
        )


# ============================================================
# IMPORTANT:
# THE PARAMETER HEAD RECEIVES 16 FEATURES, NOT THE ORIGINAL
# 6 INPUT CHANNELS.
#
# Therefore, this section examines the COMPLETE convolution
# weights and identifies which of the 16 learned feature maps
# have the strongest influence on each RGB parameter.
# ============================================================

print()
print("=" * 70)
print("16 LEARNED FEATURE MAP CONTRIBUTIONS")
print("=" * 70)

for output_channel, output_name in enumerate(
    ["R parameter", "G parameter", "B parameter"]
):

    print()
    print(output_name)

    output_weights = weights[output_channel]

    feature_strength = (
        output_weights
        .abs()
        .mean(dim=(1, 2))
    )

    sorted_indices = torch.argsort(
        feature_strength,
        descending=True
    )

    for rank, feature_index in enumerate(
        sorted_indices.tolist(),
        start=1
    ):

        value = feature_strength[feature_index].item()

        print(
            f"{rank:2d}. "
            f"Feature {feature_index:2d}"
            f"  abs-mean={value:.8f}"
        )


# ============================================================
# DIRECT COMPARISON OF RGB OUTPUT WEIGHTS
# ============================================================

print()
print("=" * 70)
print("RGB OUTPUT WEIGHT COMPARISON")
print("=" * 70)

r_weights = weights[0]
g_weights = weights[1]
b_weights = weights[2]

print()
print(
    "Mean absolute weight:"
)

print(
    f"R : {r_weights.abs().mean().item():.8f}"
)

print(
    f"G : {g_weights.abs().mean().item():.8f}"
)

print(
    f"B : {b_weights.abs().mean().item():.8f}"
)

print()
print(
    "Total absolute weight:"
)

print(
    f"R : {r_weights.abs().sum().item():.8f}"
)

print(
    f"G : {g_weights.abs().sum().item():.8f}"
)

print(
    f"B : {b_weights.abs().sum().item():.8f}"
)


# ============================================================
# B VS R / G WEIGHT DIFFERENCES
# ============================================================

print()
print("=" * 70)
print("BLUE VS RED/GREEN WEIGHT DIFFERENCE")
print("=" * 70)

blue_minus_red = (
    b_weights.abs() - r_weights.abs()
)

blue_minus_green = (
    b_weights.abs() - g_weights.abs()
)

print()
print("B - R absolute-weight difference")

print(
    f"Mean : {blue_minus_red.mean().item():+.8f}"
)

print(
    f"Min  : {blue_minus_red.min().item():+.8f}"
)

print(
    f"Max  : {blue_minus_red.max().item():+.8f}"
)

print()
print("B - G absolute-weight difference")

print(
    f"Mean : {blue_minus_green.mean().item():+.8f}"
)

print(
    f"Min  : {blue_minus_green.min().item():+.8f}"
)

print(
    f"Max  : {blue_minus_green.max().item():+.8f}"
)


# ============================================================
# SIGNED WEIGHT STATISTICS
# ============================================================

print()
print("=" * 70)
print("SIGNED WEIGHT STATISTICS")
print("=" * 70)

for channel, name in enumerate(["R", "G", "B"]):

    channel_weights = weights[channel]

    positive = (
        channel_weights > 0
    ).sum().item()

    negative = (
        channel_weights < 0
    ).sum().item()

    zero = (
        channel_weights == 0
    ).sum().item()

    total = channel_weights.numel()

    print()
    print(f"{name} OUTPUT")

    print(
        f"Positive weights : "
        f"{positive} "
        f"({100.0 * positive / total:.2f}%)"
    )

    print(
        f"Negative weights : "
        f"{negative} "
        f"({100.0 * negative / total:.2f}%)"
    )

    print(
        f"Zero weights     : "
        f"{zero} "
        f"({100.0 * zero / total:.2f}%)"
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)