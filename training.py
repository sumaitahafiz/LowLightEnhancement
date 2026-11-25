import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

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

IMAGE_SIZE = 256
BATCH_SIZE = 1

# Small sanity-check run
NUM_EPOCHS = 2
MAX_STEPS_PER_EPOCH = 5

LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# Model and pipeline
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
# Save initial model parameters
# ============================================================

initial_parameters = {
    name: parameter.detach().clone()
    for name, parameter in model.named_parameters()
}


# ============================================================
# Training
# ============================================================

print("=" * 60)
print("LOW-LIGHT ENHANCEMENT TRAINING SANITY CHECK")
print("=" * 60)

print(f"Device       : {DEVICE}")
print(f"Dataset size : {len(dataset)}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {NUM_EPOCHS}")
print(f"Steps/epoch  : {MAX_STEPS_PER_EPOCH}")
print(f"Learning rate: {LEARNING_RATE}")
print("=" * 60)


model.train()

for epoch in range(NUM_EPOCHS):

    print()
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 60)

    for step, batch in enumerate(loader):

        if step >= MAX_STEPS_PER_EPOCH:
            break

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = batch["image"].to(DEVICE)

        # ----------------------------------------------------
        # This sanity test uses batch size 1.
        # Extract the single CHW image for feature analysis.
        # ----------------------------------------------------

        single_image = image[0]

        # ----------------------------------------------------
        # Analysis feature generation
        # ----------------------------------------------------

        features = create_analysis_features(single_image)

        features = features.unsqueeze(0).to(DEVICE)

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

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        losses = loss_function(
            rgb,
            refined,
            illumination
        )

        total_loss = losses["total"]

        # ----------------------------------------------------
        # Check for invalid loss
        # ----------------------------------------------------

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite loss detected at "
                f"epoch {epoch + 1}, step {step + 1}"
            )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        total_loss.backward()

        # ----------------------------------------------------
        # Gradient check
        # ----------------------------------------------------

        gradient_norm = 0.0

        for parameter in model.parameters():

            if parameter.grad is not None:
                gradient_norm += parameter.grad.detach().norm().item() ** 2

        gradient_norm = gradient_norm ** 0.5

        if not torch.isfinite(
            torch.tensor(gradient_norm)
        ):
            raise RuntimeError(
                f"Invalid gradient detected at "
                f"epoch {epoch + 1}, step {step + 1}"
            )

        # ----------------------------------------------------
        # Update model
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"Step {step + 1}/{MAX_STEPS_PER_EPOCH} | "
            f"Total: {losses['total'].item():.6f} | "
            f"Illum: {losses['illumination'].item():.6f} | "
            f"Exposure: {losses['exposure'].item():.6f} | "
            f"Spatial: {losses['spatial'].item():.6f} | "
            f"Color: {losses['color'].item():.6f} | "
            f"Noise: {losses['noise'].item():.6f} | "
            f"Edge: {losses['edge'].item():.6f} | "
            f"Grad: {gradient_norm:.6f}"
        )


# ============================================================
# Verify that model parameters changed
# ============================================================

parameter_change_detected = False

for name, parameter in model.named_parameters():

    change = torch.mean(
        torch.abs(
            parameter.detach() - initial_parameters[name]
        )
    ).item()

    print(
        f"Parameter change - {name}: {change:.10f}"
    )

    if change > 0.0:
        parameter_change_detected = True


print()
print("=" * 60)

if parameter_change_detected:
    print("MODEL PARAMETERS UPDATED: YES")
else:
    print("MODEL PARAMETERS UPDATED: NO")

print("=" * 60)

if not parameter_change_detected:
    raise RuntimeError(
        "Training completed but model parameters did not change."
    )

print("TRAINING SANITY CHECK COMPLETED SUCCESSFULLY")
print("=" * 60)