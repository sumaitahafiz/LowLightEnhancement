import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import time

import torch
from torch.utils.data import DataLoader

from data.dataset import ExDarkDataset
from preprocessing.fusion import create_analysis_features
from model import LightweightZeroReferenceCNN
from enhancement import AdaptiveEnhancement
from refinement import ImageRefinement
from losses.total_loss import TotalLoss


# ============================================================
# Configuration
# ============================================================

EXDARK_PATH = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\ExDark\ExDark"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

HISTORY_PATH = CHECKPOINT_DIR / "training_history_corrected.txt"

IMAGE_SIZE = 256
BATCH_SIZE = 1

# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

# Start with 1 epoch for validation of the corrected objective.
# After checking the result, this can be increased to 5.
TOTAL_EPOCHS = 5
LEARNING_RATE = 1e-4

# ------------------------------------------------------------
# IMPORTANT:
# Do NOT resume from the old checkpoint.
#
# The previous checkpoints were trained using:
#     loss_function(rgb, refined, illumination)
#
# The corrected training objective uses both:
#     loss_function(rgb, enhanced, illumination)
#     loss_function(rgb, refined, illumination)
#
# Therefore we start this corrected training from scratch.
# ------------------------------------------------------------
RESUME_CHECKPOINT = (
    CHECKPOINT_DIR / "corrected_epoch_1.pth"
)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Dataset
# ============================================================

dataset = ExDarkDataset(
    root_dir=EXDARK_PATH,
    image_size=IMAGE_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ============================================================
# Model
# ============================================================

model = LightweightZeroReferenceCNN().to(DEVICE)

enhancement = AdaptiveEnhancement().to(DEVICE)

refinement = ImageRefinement().to(DEVICE)

loss_function = TotalLoss().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Resume checkpoint
# ============================================================

start_epoch = 0

if RESUME_CHECKPOINT is not None:

    if RESUME_CHECKPOINT.exists():

        print("=" * 70)
        print("CHECKPOINT FOUND")
        print("=" * 70)

        print(f"Loading: {RESUME_CHECKPOINT}")

        checkpoint = torch.load(
            RESUME_CHECKPOINT,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"]

        print(
            f"Resuming from completed epoch: "
            f"{start_epoch}"
        )

        print("=" * 70)

    else:

        print("=" * 70)
        print("CHECKPOINT NOT FOUND")
        print("Starting training from scratch.")
        print("=" * 70)

else:

    print("=" * 70)
    print("CORRECTED TRAINING")
    print("=" * 70)

    print("Starting training from scratch.")

    print(
        "Loss is applied to both the enhanced "
        "and refined stages."
    )

    print("=" * 70)


# ============================================================
# Training information
# ============================================================

print()

print("=" * 70)
print("LOW-LIGHT ENHANCEMENT — CORRECTED TRAINING")
print("=" * 70)

print(f"Device        : {DEVICE}")
print(f"Dataset size  : {len(dataset)}")
print(f"Image size    : {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Start epoch   : {start_epoch + 1}")
print(f"Final epoch   : {TOTAL_EPOCHS}")
print(f"Learning rate : {LEARNING_RATE}")

print(
    "Loss stages   : Enhanced + Refined"
)

print("=" * 70)


# ============================================================
# Training history
# ============================================================

if not HISTORY_PATH.exists():

    with open(HISTORY_PATH, "w") as file:

        file.write(
            "epoch,total,illumination,exposure,"
            "spatial,color,noise,edge,time_minutes\n"
        )


# ============================================================
# Training
# ============================================================

training_start = time.time()

model.train()

for epoch in range(start_epoch, TOTAL_EPOCHS):

    epoch_number = epoch + 1

    epoch_start = time.time()

    total_loss_sum = 0.0
    illumination_sum = 0.0
    exposure_sum = 0.0
    spatial_sum = 0.0
    color_sum = 0.0
    noise_sum = 0.0
    edge_sum = 0.0

    num_batches = 0

    print()

    print("=" * 70)

    print(
        f"Epoch {epoch_number}/{TOTAL_EPOCHS}"
    )

    print("=" * 70)


    # ========================================================
    # Batch loop
    # ========================================================

    for step, batch in enumerate(loader):

        image = batch["image"].to(DEVICE)


        # ----------------------------------------------------
        # Batch size = 1
        # ----------------------------------------------------

        single_image = image[0]


        # ----------------------------------------------------
        # Analysis feature generation
        # ----------------------------------------------------

        features = create_analysis_features(
            single_image
        )

        features = features.unsqueeze(0).to(DEVICE)


        # ----------------------------------------------------
        # Separate RGB and illumination
        # ----------------------------------------------------

        rgb = features[:, 0:3, :, :]

        illumination = features[:, 3:4, :, :]


        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        parameters = model(features)


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


        # ====================================================
        # Zero-reference losses
        # ====================================================

        # ----------------------------------------------------
        # Losses on Enhanced Image
        # ----------------------------------------------------

        enhanced_losses = loss_function(
            rgb,
            enhanced,
            illumination
        )


        # ----------------------------------------------------
        # Losses on Final Refined Image
        # ----------------------------------------------------

        refined_losses = loss_function(
            rgb,
            refined,
            illumination
        )


        # ----------------------------------------------------
        # Combined objective
        #
        # Equal contribution from:
        #   1. Enhanced Image
        #   2. Refined Image
        #
        # The averaging keeps the loss scale comparable
        # with the previous training objective.
        # ----------------------------------------------------

        total_loss = (
            0.5 * enhanced_losses["total"]
            +
            0.5 * refined_losses["total"]
        )


        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if not torch.isfinite(total_loss):

            raise RuntimeError(
                f"Non-finite loss detected at "
                f"epoch {epoch_number}, "
                f"step {step + 1}"
            )


        # ====================================================
        # Backpropagation
        # ====================================================

        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()


        # ====================================================
        # Accumulate losses
        # ====================================================

        total_loss_sum += total_loss.item()


        illumination_sum += (
            0.5 * enhanced_losses["illumination"].item()
            +
            0.5 * refined_losses["illumination"].item()
        )


        exposure_sum += (
            0.5 * enhanced_losses["exposure"].item()
            +
            0.5 * refined_losses["exposure"].item()
        )


        spatial_sum += (
            0.5 * enhanced_losses["spatial"].item()
            +
            0.5 * refined_losses["spatial"].item()
        )


        color_sum += (
            0.5 * enhanced_losses["color"].item()
            +
            0.5 * refined_losses["color"].item()
        )


        noise_sum += (
            0.5 * enhanced_losses["noise"].item()
            +
            0.5 * refined_losses["noise"].item()
        )


        edge_sum += (
            0.5 * enhanced_losses["edge"].item()
            +
            0.5 * refined_losses["edge"].item()
        )


        num_batches += 1


        # ====================================================
        # Progress
        # ====================================================

        if (step + 1) % 100 == 0:

            print(
                f"Step {step + 1:4d}/{len(loader)} | "
                f"Loss: {total_loss.item():.6f}"
            )


    # ========================================================
    # Epoch averages
    # ========================================================

    avg_total = (
        total_loss_sum / num_batches
    )

    avg_illumination = (
        illumination_sum / num_batches
    )

    avg_exposure = (
        exposure_sum / num_batches
    )

    avg_spatial = (
        spatial_sum / num_batches
    )

    avg_color = (
        color_sum / num_batches
    )

    avg_noise = (
        noise_sum / num_batches
    )

    avg_edge = (
        edge_sum / num_batches
    )


    epoch_time = (
        time.time() - epoch_start
    )

    epoch_minutes = epoch_time / 60.0


    # ========================================================
    # Print epoch summary
    # ========================================================

    print()

    print("Epoch summary")

    print("-" * 70)

    print(
        f"Average total loss  : "
        f"{avg_total:.6f}"
    )

    print(
        f"Average illumination : "
        f"{avg_illumination:.6f}"
    )

    print(
        f"Average exposure     : "
        f"{avg_exposure:.6f}"
    )

    print(
        f"Average spatial      : "
        f"{avg_spatial:.6f}"
    )

    print(
        f"Average color        : "
        f"{avg_color:.6f}"
    )

    print(
        f"Average noise        : "
        f"{avg_noise:.6f}"
    )

    print(
        f"Average edge         : "
        f"{avg_edge:.6f}"
    )

    print(
        f"Epoch time           : "
        f"{epoch_minutes:.2f} minutes"
    )


    # ========================================================
    # Save history
    # ========================================================

    with open(HISTORY_PATH, "a") as file:

        file.write(
            f"{epoch_number},"
            f"{avg_total:.8f},"
            f"{avg_illumination:.8f},"
            f"{avg_exposure:.8f},"
            f"{avg_spatial:.8f},"
            f"{avg_color:.8f},"
            f"{avg_noise:.8f},"
            f"{avg_edge:.8f},"
            f"{epoch_minutes:.4f}\n"
        )


    # ========================================================
    # Save checkpoint
    # ========================================================

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"corrected_epoch_{epoch_number}.pth"
    )


    torch.save(
        {
            "epoch": epoch_number,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss":
                avg_total,

            "learning_rate":
                LEARNING_RATE,
        },
        checkpoint_path
    )


    print(
        f"Checkpoint saved: "
        f"{checkpoint_path}"
    )


# ============================================================
# Finish
# ============================================================

total_minutes = (
    (time.time() - training_start) / 60.0
)


print()

print("=" * 70)

print("CORRECTED TRAINING COMPLETED")

print("=" * 70)

print(
    f"Total training time: "
    f"{total_minutes:.2f} minutes"
)

print(
    f"Training history  : "
    f"{HISTORY_PATH}"
)

print("=" * 70)