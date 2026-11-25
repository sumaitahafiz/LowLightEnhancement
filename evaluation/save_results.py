import sys
import gc

from pathlib import Path

import torch
from PIL import Image

from torchvision import transforms

import matplotlib.pyplot as plt


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from model import LightweightZeroReferenceCNN
from preprocessing.fusion import create_analysis_features
from enhancement import AdaptiveEnhancement
from refinement import ImageRefinement


# ============================================================
# Paths
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "corrected_epoch_5.pth"
)

LOL_V1_ROOT = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\LOLdataset\eval15"
)

LOL_V2_ROOT = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\LOL-v2"
    r"\LOL-v2\Real_captured\Test"
)

EXDARK_ROOT = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\ExDark\ExDark"
)

RESULTS_DIR = (
    PROJECT_ROOT / "evaluation" / "results"
)

IMAGE_SIZE = 256

DEVICE = torch.device("cpu")

# Process sequentially in manageable chunks
CHUNK_SIZE = 100


# ============================================================
# Preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
])


def load_image(path):

    with Image.open(path) as image:

        image = image.convert("RGB")

        image = transform(image)

    return image


# ============================================================
# Load model
# ============================================================

def load_model():

    model = LightweightZeroReferenceCNN().to(
        DEVICE
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "model" in checkpoint:

        state_dict = checkpoint["model"]

    else:

        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    return model


# ============================================================
# Enhancement pipeline
# ============================================================

def enhance_image(
    model,
    enhancement,
    refinement,
    image
):

    # --------------------------------------------------------
    # Create analysis features
    # --------------------------------------------------------

    features = create_analysis_features(
        image
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb = (
        features[0:3]
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Illumination map
    # --------------------------------------------------------

    illumination = (
        features[3:4]
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Complete feature tensor
    # --------------------------------------------------------

    features = (
        features
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Lightweight CNN
    # --------------------------------------------------------

    parameters = model(features)

    # --------------------------------------------------------
    # Adaptive enhancement
    # --------------------------------------------------------

    enhanced = enhancement(
        rgb,
        illumination,
        parameters
    )

    # --------------------------------------------------------
    # Refinement
    # --------------------------------------------------------

    refined = refinement(
        rgb,
        enhanced
    )

    # Return both stages
    return (
        enhanced.squeeze(0),
        refined.squeeze(0)
    )


# ============================================================
# Tensor → NumPy
# ============================================================

def tensor_to_numpy(image):

    image = image.detach().cpu()

    image = torch.clamp(
        image,
        0.0,
        1.0
    )

    return image.permute(
        1,
        2,
        0
    ).numpy()


# ============================================================
# Save LOL comparison
# ============================================================

def save_lol_comparison(
    low_path,
    reference_path,
    output_path,
    title
):

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    low = load_image(
        low_path
    )

    reference = load_image(
        reference_path
    )

    # --------------------------------------------------------
    # Enhancement
    # --------------------------------------------------------

    with torch.inference_mode():

        enhanced, refined = enhance_image(
            model,
            enhancement,
            refinement,
            low
        )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    low_np = tensor_to_numpy(
        low
    )

    enhanced_np = tensor_to_numpy(
        enhanced
    )

    refined_np = tensor_to_numpy(
        refined
    )

    reference_np = tensor_to_numpy(
        reference
    )

    # --------------------------------------------------------
    # Create four-image comparison
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    # Low-light input
    axes[0].imshow(
        low_np
    )

    axes[0].set_title(
        "Low-Light Input"
    )

    axes[0].axis("off")

    # Adaptive enhanced
    axes[1].imshow(
        enhanced_np
    )

    axes[1].set_title(
        "Adaptive Enhanced"
    )

    axes[1].axis("off")

    # Final refined
    axes[2].imshow(
        refined_np
    )

    axes[2].set_title(
        "Final Refined"
    )

    axes[2].axis("off")

    # Reference
    axes[3].imshow(
        reference_np
    )

    axes[3].set_title(
        "Reference"
    )

    axes[3].axis("off")

    # Overall title
    figure.suptitle(
        title
    )

    figure.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


# ============================================================
# Save ExDark comparison
# ============================================================

def save_exdark_comparison(
    image_path,
    output_path,
    title
):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    low = load_image(
        image_path
    )

    # --------------------------------------------------------
    # Enhancement
    # --------------------------------------------------------

    with torch.inference_mode():

        enhanced, refined = enhance_image(
            model,
            enhancement,
            refinement,
            low
        )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    low_np = tensor_to_numpy(
        low
    )

    enhanced_np = tensor_to_numpy(
        enhanced
    )

    refined_np = tensor_to_numpy(
        refined
    )

    # --------------------------------------------------------
    # Create three-image comparison
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # Low-light input
    axes[0].imshow(
        low_np
    )

    axes[0].set_title(
        "ExDark Low-Light Input"
    )

    axes[0].axis("off")

    # Adaptive enhanced
    axes[1].imshow(
        enhanced_np
    )

    axes[1].set_title(
        "Adaptive Enhanced"
    )

    axes[1].axis("off")

    # Final refined
    axes[2].imshow(
        refined_np
    )

    axes[2].set_title(
        "Final Refined"
    )

    axes[2].axis("off")

    # Overall title
    figure.suptitle(
        title
    )

    figure.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


# ============================================================
# LOL-v1
# ============================================================

def save_lol_v1_results():

    low_dir = (
        LOL_V1_ROOT / "low"
    )

    high_dir = (
        LOL_V1_ROOT / "high"
    )

    output_dir = (
        RESULTS_DIR / "LOLv1"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    low_images = sorted(
        [
            path
            for path in low_dir.iterdir()
            if path.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp"
            }
        ]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SAVING FULL LOL-v1 QUALITATIVE RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Images to process: {len(low_images)}"
    )

    total = len(
        low_images
    )

    processed = 0

    # --------------------------------------------------------
    # Sequential chunks
    # --------------------------------------------------------

    for chunk_start in range(
        0,
        total,
        CHUNK_SIZE
    ):

        chunk_end = min(
            chunk_start + CHUNK_SIZE,
            total
        )

        print(
            f"\nProcessing LOL-v1 "
            f"{chunk_start + 1}-{chunk_end} "
            f"of {total}..."
        )

        for low_path in low_images[
            chunk_start:chunk_end
        ]:

            reference_path = (
                high_dir / low_path.name
            )

            if not reference_path.exists():

                print(
                    f"WARNING: Missing reference "
                    f"for {low_path.name}"
                )

                continue

            output_path = (
                output_dir
                / f"LOLv1_{low_path.stem}.png"
            )

            save_lol_comparison(
                low_path,
                reference_path,
                output_path,
                f"LOL-v1 — {low_path.name}"
            )

            processed += 1

        gc.collect()

        print(
            f"Completed: "
            f"{processed}/{total}"
        )

    print(
        f"\nLOL-v1 results saved: "
        f"{processed}"
    )


# ============================================================
# LOL-v2
# ============================================================

def save_lol_v2_results():

    low_dir = (
        LOL_V2_ROOT / "Low"
    )

    normal_dir = (
        LOL_V2_ROOT / "Normal"
    )

    output_dir = (
        RESULTS_DIR / "LOLv2"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    low_images = sorted(
        [
            path
            for path in low_dir.iterdir()
            if path.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp"
            }
        ]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SAVING FULL LOL-v2 QUALITATIVE RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Images to process: {len(low_images)}"
    )

    total = len(
        low_images
    )

    processed = 0

    # --------------------------------------------------------
    # Sequential chunks
    # --------------------------------------------------------

    for chunk_start in range(
        0,
        total,
        CHUNK_SIZE
    ):

        chunk_end = min(
            chunk_start + CHUNK_SIZE,
            total
        )

        print(
            f"\nProcessing LOL-v2 "
            f"{chunk_start + 1}-{chunk_end} "
            f"of {total}..."
        )

        for low_path in low_images[
            chunk_start:chunk_end
        ]:

            # low00690.png
            # →
            # normal00690.png

            normal_filename = (
                low_path.name.replace(
                    "low",
                    "normal",
                    1
                )
            )

            reference_path = (
                normal_dir / normal_filename
            )

            if not reference_path.exists():

                print(
                    f"WARNING: Missing reference "
                    f"for {low_path.name}"
                )

                continue

            output_path = (
                output_dir
                / f"LOLv2_{low_path.stem}.png"
            )

            save_lol_comparison(
                low_path,
                reference_path,
                output_path,
                f"LOL-v2 Real — "
                f"{low_path.name}"
            )

            processed += 1

        gc.collect()

        print(
            f"Completed: "
            f"{processed}/{total}"
        )

    print(
        f"\nLOL-v2 results saved: "
        f"{processed}"
    )


# ============================================================
# ExDark
# ============================================================

def save_exdark_results():

    output_dir = (
        RESULTS_DIR / "ExDark"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SAVING FULL ExDark QUALITATIVE RESULTS"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Find ALL ExDark images
    # --------------------------------------------------------

    image_paths = sorted(
        [
            path
            for path in EXDARK_ROOT.rglob("*")
            if path.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp"
            }
        ]
    )

    total = len(
        image_paths
    )

    print(
        f"Images to process: {total}"
    )

    processed = 0

    # --------------------------------------------------------
    # Sequential chunks
    # --------------------------------------------------------

    for chunk_start in range(
        0,
        total,
        CHUNK_SIZE
    ):

        chunk_end = min(
            chunk_start + CHUNK_SIZE,
            total
        )

        print(
            f"\nProcessing ExDark "
            f"{chunk_start + 1}-{chunk_end} "
            f"of {total}..."
        )

        for image_path in image_paths[
            chunk_start:chunk_end
        ]:

            # ------------------------------------------------
            # Preserve class information
            # ------------------------------------------------

            class_name = (
                image_path.parent.name
            )

            # ------------------------------------------------
            # Output filename
            # ------------------------------------------------

            output_name = (
                f"ExDark_{class_name}_"
                f"{image_path.stem}.png"
            )

            output_path = (
                output_dir / output_name
            )

            # ------------------------------------------------
            # Generate comparison
            # ------------------------------------------------

            save_exdark_comparison(
                image_path,
                output_path,
                f"ExDark — "
                f"{class_name} — "
                f"{image_path.name}"
            )

            processed += 1

        gc.collect()

        print(
            f"Completed: "
            f"{processed}/{total}"
        )

    print(
        f"\nExDark results saved: "
        f"{processed}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "FULL QUALITATIVE EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nCheckpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print(
        "\nDevice:"
    )

    print(
        DEVICE
    )

    # --------------------------------------------------------
    # Create exactly three result directories
    # --------------------------------------------------------

    (RESULTS_DIR / "ExDark").mkdir(
        parents=True,
        exist_ok=True
    )

    (RESULTS_DIR / "LOLv1").mkdir(
        parents=True,
        exist_ok=True
    )

    (RESULTS_DIR / "LOLv2").mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Enhancement module
    # --------------------------------------------------------

    enhancement = (
        AdaptiveEnhancement()
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Refinement module
    # --------------------------------------------------------

    refinement = (
        ImageRefinement()
        .to(DEVICE)
    )

    enhancement.eval()

    refinement.eval()

    print(
        "\nModel loaded successfully."
    )

    # --------------------------------------------------------
    # LOL-v1
    # --------------------------------------------------------

    save_lol_v1_results()

    # --------------------------------------------------------
    # LOL-v2
    # --------------------------------------------------------

    save_lol_v2_results()

    # --------------------------------------------------------
    # ExDark
    # --------------------------------------------------------

    save_exdark_results()

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FULL QUALITATIVE EVALUATION COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nResults directory:"
    )

    print(
        RESULTS_DIR
    )