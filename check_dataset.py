from pathlib import Path

# ============================================================
# DATASET ROOT
# ============================================================

DATASET_ROOT = Path(r"D:\12th Trimester\DIP\DIP Paper\Datasets")

EXDARK_ROOT = DATASET_ROOT / "ExDark" / "ExDark"
LOL_V1_ROOT = DATASET_ROOT / "LOLdataset"
LOL_V2_ROOT = DATASET_ROOT / "LOL-v2" / "LOL-v2" / "Real_captured"


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def count_images(folder):
    """Count image files recursively inside a folder."""
    if not folder.exists():
        return 0

    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def show_folder(folder, name):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Path:")
    print(folder)

    print("\nExists:", folder.exists())

    if not folder.exists():
        return

    print("\nImage count:", count_images(folder))

    print("\nSubfolders:")

    for item in sorted(folder.iterdir()):
        if item.is_dir():
            print(
                f"  {item.name:<20} "
                f"{count_images(item)} images"
            )


# ============================================================
# CHECK DATASETS
# ============================================================

print("\nDATASET ROOT")
print("=" * 60)
print(DATASET_ROOT)
print("Exists:", DATASET_ROOT.exists())


show_folder(
    EXDARK_ROOT,
    "ExDark"
)

show_folder(
    LOL_V1_ROOT / "our485",
    "LOL-v1 / our485"
)

show_folder(
    LOL_V1_ROOT / "eval15",
    "LOL-v1 / eval15"
)

show_folder(
    LOL_V2_ROOT / "Train",
    "LOL-v2 Real / Train"
)

show_folder(
    LOL_V2_ROOT / "Test",
    "LOL-v2 Real / Test"
)