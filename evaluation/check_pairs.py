from pathlib import Path


# ============================================================
# Dataset paths
# ============================================================

LOL_V1_EVAL = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\LOLdataset\eval15"
)

LOL_V2_TEST = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\LOL-v2\LOL-v2\Real_captured\Test"
)


# ============================================================
# Helper
# ============================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    return sorted(
        [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def show_structure(root):
    print("\n" + "=" * 70)
    print(f"ROOT: {root}")
    print("=" * 70)

    if not root.exists():
        print("ERROR: Folder does not exist.")
        return

    print("\nImmediate contents:")

    for item in sorted(root.iterdir()):
        if item.is_dir():
            print(f"  [DIR ] {item.name}")
        else:
            print(f"  [FILE] {item.name}")

    images = list_images(root)

    print(f"\nTotal images found recursively: {len(images)}")

    print("\nFirst 20 images:")

    for image in images[:20]:
        print(f"  {image.relative_to(root)}")


# ============================================================
# LOL-v1
# ============================================================

show_structure(LOL_V1_EVAL)


# ============================================================
# LOL-v2
# ============================================================

show_structure(LOL_V2_TEST)


print("\n" + "=" * 70)
print("PAIRING INSPECTION COMPLETED")
print("=" * 70)