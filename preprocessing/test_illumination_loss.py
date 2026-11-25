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
from losses.illumination_loss import IlluminationLoss


print("=" * 60)
print("ILLUMINATION LOSS PIPELINE TEST")
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
# 2. Create analysis features
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
# 7. Estimate illumination of refined image
# ------------------------------------------------------------

refined_illumination = estimate_illumination(
    refined[0]
)

refined_illumination = refined_illumination.unsqueeze(0)

print(
    "Refined illumination :",
    refined_illumination.shape
)


# ------------------------------------------------------------
# 8. Calculate illumination loss
# ------------------------------------------------------------

loss_function = IlluminationLoss()

loss = loss_function(
    refined,
    refined_illumination
)

print()
print("Illumination loss    :", loss.item())


# ------------------------------------------------------------
# 9. Check gradients
# ------------------------------------------------------------

refined_for_gradient = refined.clone().detach()
refined_for_gradient.requires_grad_(True)

refined_illumination_2 = estimate_illumination(
    refined_for_gradient[0]
)

refined_illumination_2 = (
    refined_illumination_2.unsqueeze(0)
)

gradient_loss = loss_function(
    refined_for_gradient,
    refined_illumination_2
)

gradient_loss.backward()

print(
    "Gradient available  :",
    refined_for_gradient.grad is not None
)

print("=" * 60)
print("ILLUMINATION LOSS PIPELINE TEST COMPLETED")
print("=" * 60)