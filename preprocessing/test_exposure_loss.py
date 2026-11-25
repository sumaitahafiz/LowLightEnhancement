import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


import torch

from data.dataset import ExDarkDataset
from preprocessing.fusion import create_analysis_features
from preprocessing.illumination import estimate_illumination
from model import LightweightZeroReferenceCNN
from enhancement import AdaptiveEnhancement
from refinement import ImageRefinement
from losses.exposure_loss import ExposureLoss


print("=" * 60)
print("CONTRAST / EXPOSURE LOSS PIPELINE TEST")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load ExDark
# ------------------------------------------------------------

EXDARK_ROOT = (
    r"D:\12th Trimester\DIP\DIP Paper"
    r"\Datasets\ExDark\ExDark"
)

dataset = ExDarkDataset(EXDARK_ROOT)

sample = dataset[0]

image = sample["image"]
image_path = sample["path"]

print("Image path          :", image_path)
print("Original image shape:", image.shape)


# ------------------------------------------------------------
# 2. Feature fusion
# ------------------------------------------------------------

features = create_analysis_features(image)

print("Feature tensor shape:", features.shape)


# ------------------------------------------------------------
# 3. Prepare batches
# ------------------------------------------------------------

rgb = image.unsqueeze(0)
features = features.unsqueeze(0)

illumination = estimate_illumination(image)
illumination = illumination.unsqueeze(0)

print("RGB batch shape      :", rgb.shape)
print("Illumination shape   :", illumination.shape)


# ------------------------------------------------------------
# 4. CNN
# ------------------------------------------------------------

model = LightweightZeroReferenceCNN()
model.eval()

with torch.no_grad():
    parameters = model(features)

print("CNN parameter shape  :", parameters.shape)


# ------------------------------------------------------------
# 5. Adaptive enhancement
# ------------------------------------------------------------

enhancer = AdaptiveEnhancement()
enhancer.eval()

with torch.no_grad():
    enhanced = enhancer(
        rgb,
        illumination,
        parameters
    )

print("Enhanced image shape :", enhanced.shape)


# ------------------------------------------------------------
# 6. Refinement
# ------------------------------------------------------------

refiner = ImageRefinement()
refiner.eval()

with torch.no_grad():
    refined = refiner(
        rgb,
        enhanced
    )

print("Refined image shape  :", refined.shape)


# ------------------------------------------------------------
# 7. Exposure loss
# ------------------------------------------------------------

loss_function = ExposureLoss(
    target_mean=0.6,
    patch_size=16
)

loss = loss_function(refined)

print()
print("Exposure loss       :", loss.item())


# ------------------------------------------------------------
# 8. Gradient test
# ------------------------------------------------------------

refined_for_gradient = refined.clone().detach()
refined_for_gradient.requires_grad_(True)

gradient_loss = loss_function(
    refined_for_gradient
)

gradient_loss.backward()

print(
    "Gradient available  :",
    refined_for_gradient.grad is not None
)

print("=" * 60)
print("EXPOSURE LOSS PIPELINE TEST COMPLETED")
print("=" * 60)