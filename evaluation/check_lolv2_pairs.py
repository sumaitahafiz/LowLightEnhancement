from pathlib import Path


TEST_ROOT = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\LOL-v2\LOL-v2\Real_captured\Test"
)

LOW_DIR = TEST_ROOT / "Low"
NORMAL_DIR = TEST_ROOT / "Normal"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_images(folder):
    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


low_images = get_images(LOW_DIR)
normal_images = get_images(NORMAL_DIR)

print("=" * 70)
print("LOL-v2 REAL TEST PAIRING INSPECTION")
print("=" * 70)

print(f"\nLow images    : {len(low_images)}")
print(f"Normal images : {len(normal_images)}")

print("\nFirst 30 Low filenames:")
for image in low_images[:30]:
    print(f"  {image.name}")

print("\nFirst 30 Normal filenames:")
for image in normal_images[:30]:
    print(f"  {image.name}")

print("\n" + "=" * 70)
print("PAIRING CHECK")
print("=" * 70)

if len(low_images) != len(normal_images):
    print("WARNING: Different numbers of Low and Normal images.")
else:
    print("Counts match.")

print("\nPotential positional pairs:")

for low, normal in zip(low_images[:30], normal_images[:30]):
    print(f"  {low.name:<20} <-> {normal.name}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)