from pathlib import Path

from dataset import ExDarkDataset


# ============================================================
# EXDARK PATH
# ============================================================

EXDARK_PATH = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\ExDark\ExDark"
)


# ============================================================
# CREATE DATASET
# ============================================================

dataset = ExDarkDataset(
    root_dir=EXDARK_PATH,
    image_size=256,
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("=" * 60)
print("ExDark Dataset Test")
print("=" * 60)

print("Number of images:", len(dataset))


# ============================================================
# LOAD FIRST IMAGE
# ============================================================

sample = dataset[0]

image = sample["image"]
path = sample["path"]

print("\nFirst image:")
print("Path:", path)
print("Tensor shape:", image.shape)
print("Data type:", image.dtype)
print("Minimum:", image.min().item())
print("Maximum:", image.max().item())