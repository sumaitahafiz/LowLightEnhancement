import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)
import torch
import matplotlib.pyplot as plt

from data.dataset import ExDarkDataset
from preprocessing.fusion import create_analysis_features
from preprocessing.illumination import estimate_illumination
from model import LightweightZeroReferenceCNN
from enhancement import AdaptiveEnhancement
from refinement import ImageRefinement


print("=" * 60)
print("REFINEMENT PIPELINE TEST")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

EXDARK_ROOT = r"D:\12th Trimester\DIP\DIP Paper\Datasets\ExDark\ExDark"

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
# 4. Lightweight CNN
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
print("Refined minimum      :", refined.min().item())
print("Refined maximum      :", refined.max().item())


# ------------------------------------------------------------
# 7. Visualization
# ------------------------------------------------------------

original_display = rgb[0].permute(1, 2, 0).numpy()
enhanced_display = enhanced[0].permute(1, 2, 0).numpy()
refined_display = refined[0].permute(1, 2, 0).numpy()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_display)
plt.title("Original Low-Light Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(enhanced_display)
plt.title("Adaptive Enhancement")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(refined_display)
plt.title("Refined Image")
plt.axis("off")

plt.tight_layout()
plt.show()


print("=" * 60)
print("REFINEMENT PIPELINE TEST COMPLETED")
print("=" * 60)