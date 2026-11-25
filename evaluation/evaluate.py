import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import pyiqa


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
    PROJECT_ROOT
    / "checkpoints"
    / "corrected_epoch_5.pth"
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


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 256

DEVICE = torch.device("cpu")

# Full LOL-v1 and LOL-v2 evaluation
LOL_V1_LIMIT = None
LOL_V2_LIMIT = None

# Full ExDark evaluation
EXDARK_LIMIT = None

# Progress is reported after every chunk
CHUNK_SIZE = 100


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def load_image(path):
    image = Image.open(path).convert("RGB")
    image = transform(image)
    return image


# ============================================================
# Load trained model
# ============================================================

def load_model():

    model = LightweightZeroReferenceCNN().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    # Handle checkpoint dictionary
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

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
    """
    image:
        Tensor [3, H, W]

    Returns:
        refined image [3, H, W]
    """

    # --------------------------------------------------------
    # Create analysis features
    # --------------------------------------------------------

    features = create_analysis_features(image)

    # --------------------------------------------------------
    # Separate RGB and illumination
    # --------------------------------------------------------

    rgb = features[0:3].unsqueeze(0).to(DEVICE)

    illumination = features[3:4].unsqueeze(0).to(DEVICE)

    features = features.unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # CNN
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

    return refined.squeeze(0)


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

    image = image.permute(
        1,
        2,
        0
    ).numpy()

    return image


# ============================================================
# Reference-based metrics
# ============================================================

def calculate_metrics(
    enhanced,
    reference
):

    psnr = peak_signal_noise_ratio(
        reference,
        enhanced,
        data_range=1.0
    )

    ssim = structural_similarity(
        reference,
        enhanced,
        channel_axis=2,
        data_range=1.0
    )

    return psnr, ssim


# ============================================================
# ExDark no-reference metrics
# ============================================================

def calculate_no_reference_metrics(image):
    """
    Calculate basic image statistics used for
    no-reference low-light enhancement analysis.

    image:
        Tensor [3, H, W], range [0, 1]

    Returns:
        brightness
        contrast
        noise
        edge_strength
    """

    image = torch.clamp(
        image,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Convert RGB to grayscale
    # --------------------------------------------------------

    gray = (
        0.299 * image[0:1] +
        0.587 * image[1:2] +
        0.114 * image[2:3]
    )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = gray.mean().item()

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast = gray.std().item()

    # --------------------------------------------------------
    # Noise / high-frequency variation
    # --------------------------------------------------------

    gray_batch = gray.unsqueeze(0)

    smoothed = torch.nn.functional.avg_pool2d(
        gray_batch,
        kernel_size=3,
        stride=1,
        padding=1
    )

    noise = torch.mean(
        torch.abs(
            gray_batch - smoothed
        )
    ).item()

    # --------------------------------------------------------
    # Edge / detail strength
    # --------------------------------------------------------

    sobel_x = torch.tensor(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        dtype=image.dtype,
        device=image.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],
        dtype=image.dtype,
        device=image.device
    ).view(1, 1, 3, 3)

    grad_x = torch.nn.functional.conv2d(
        gray_batch,
        sobel_x,
        padding=1
    )

    grad_y = torch.nn.functional.conv2d(
        gray_batch,
        sobel_y,
        padding=1
    )

    edge_strength = torch.sqrt(
        grad_x ** 2 +
        grad_y ** 2 +
        1e-6
    ).mean().item()

    return (
        brightness,
        contrast,
        noise,
        edge_strength
    )


# ============================================================
# ExDark no-reference evaluation
# ============================================================

def evaluate_exdark_no_reference(
    model,
    enhancement,
    refinement
):

    # --------------------------------------------------------
    # Find all ExDark images
    # --------------------------------------------------------

    image_paths = sorted(
        EXDARK_ROOT.rglob("*")
    )

    image_paths = [
        path
        for path in image_paths
        if path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp"
        }
    ]

    if EXDARK_LIMIT is not None:
        image_paths = image_paths[:EXDARK_LIMIT]

    total_images = len(image_paths)

    if total_images == 0:
        print("\nNo ExDark images found.")
        return

    # --------------------------------------------------------
    # Initialize NIQE and BRISQUE
    # --------------------------------------------------------

    print("\nInitializing NIQE and BRISQUE...")

    niqe_metric = pyiqa.create_metric(
        "niqe",
        device=DEVICE
    )

    brisque_metric = pyiqa.create_metric(
        "brisque",
        device=DEVICE
    )

    niqe_metric.eval()
    brisque_metric.eval()

    # --------------------------------------------------------
    # Running totals
    # --------------------------------------------------------

    sum_original_niqe = 0.0
    sum_refined_niqe = 0.0

    sum_original_brisque = 0.0
    sum_refined_brisque = 0.0

    sum_original_brightness = 0.0
    sum_refined_brightness = 0.0

    sum_original_contrast = 0.0
    sum_refined_contrast = 0.0

    sum_original_noise = 0.0
    sum_refined_noise = 0.0

    sum_original_edges = 0.0
    sum_refined_edges = 0.0

    processed = 0

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXDARK FULL NO-REFERENCE EVALUATION")
    print("=" * 70)

    print(
        "Note: ExDark does not contain a predefined "
        "held-out test split in this project."
    )

    print(
        f"Images to analyze: {total_images}"
    )

    print(
        f"Processing chunk size: {CHUNK_SIZE}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Process in chunks
    # --------------------------------------------------------

    for chunk_start in range(
        0,
        total_images,
        CHUNK_SIZE
    ):

        chunk_end = min(
            chunk_start + CHUNK_SIZE,
            total_images
        )

        chunk_paths = image_paths[
            chunk_start:chunk_end
        ]

        print(
            f"\nProcessing images "
            f"{chunk_start + 1}-{chunk_end} "
            f"of {total_images}..."
        )

        # ----------------------------------------------------
        # Process images in current chunk
        # ----------------------------------------------------

        for image_path in chunk_paths:

            image = load_image(image_path)

            with torch.inference_mode():

                refined = enhance_image(
                    model,
                    enhancement,
                    refinement,
                    image
                )

                original_batch = (
                    image
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                refined_batch = (
                    refined
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                # ------------------------------------------------
                # NIQE
                # ------------------------------------------------

                original_niqe_value = (
                    niqe_metric(
                        original_batch
                    ).item()
                )

                refined_niqe_value = (
                    niqe_metric(
                        refined_batch
                    ).item()
                )

                # ------------------------------------------------
                # BRISQUE
                # ------------------------------------------------

                original_brisque_value = (
                    brisque_metric(
                        original_batch
                    ).item()
                )

                refined_brisque_value = (
                    brisque_metric(
                        refined_batch
                    ).item()
                )

            # ------------------------------------------------
            # Basic statistics
            # ------------------------------------------------

            (
                brightness_original,
                contrast_original,
                noise_original,
                edge_original
            ) = calculate_no_reference_metrics(
                image
            )

            (
                brightness_refined,
                contrast_refined,
                noise_refined,
                edge_refined
            ) = calculate_no_reference_metrics(
                refined
            )

            # ------------------------------------------------
            # Add to running totals
            # ------------------------------------------------

            sum_original_niqe += (
                original_niqe_value
            )

            sum_refined_niqe += (
                refined_niqe_value
            )

            sum_original_brisque += (
                original_brisque_value
            )

            sum_refined_brisque += (
                refined_brisque_value
            )

            sum_original_brightness += (
                brightness_original
            )

            sum_refined_brightness += (
                brightness_refined
            )

            sum_original_contrast += (
                contrast_original
            )

            sum_refined_contrast += (
                contrast_refined
            )

            sum_original_noise += (
                noise_original
            )

            sum_refined_noise += (
                noise_refined
            )

            sum_original_edges += (
                edge_original
            )

            sum_refined_edges += (
                edge_refined
            )

            processed += 1

        # ----------------------------------------------------
        # Chunk progress
        # ----------------------------------------------------

        print(
            f"Completed: {processed}/{total_images} "
            f"images"
        )

        print(
            f"Progress: "
            f"{processed / total_images * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Final averages
    # --------------------------------------------------------

    if processed == 0:
        print("\nNo ExDark images were evaluated.")
        return

    print("\n" + "=" * 70)
    print("EXDARK FULL-DATASET NO-REFERENCE AVERAGES")
    print("=" * 70)

    print(
        f"Images analyzed: {processed}"
    )

    print(
        f"NIQE       : "
        f"{sum_original_niqe / processed:.4f}"
        f" -> "
        f"{sum_refined_niqe / processed:.4f}"
    )

    print(
        f"BRISQUE    : "
        f"{sum_original_brisque / processed:.4f}"
        f" -> "
        f"{sum_refined_brisque / processed:.4f}"
    )

    print(
        f"Brightness : "
        f"{sum_original_brightness / processed:.4f}"
        f" -> "
        f"{sum_refined_brightness / processed:.4f}"
    )

    print(
        f"Contrast   : "
        f"{sum_original_contrast / processed:.4f}"
        f" -> "
        f"{sum_refined_contrast / processed:.4f}"
    )

    print(
        f"Noise      : "
        f"{sum_original_noise / processed:.6f}"
        f" -> "
        f"{sum_refined_noise / processed:.6f}"
    )

    print(
        f"Edge/detail: "
        f"{sum_original_edges / processed:.6f}"
        f" -> "
        f"{sum_refined_edges / processed:.6f}"
    )

    print("=" * 70)


# ============================================================
# LOL-v1 evaluation
# ============================================================

def evaluate_lol_v1(
    model,
    enhancement,
    refinement
):

    low_dir = LOL_V1_ROOT / "low"
    high_dir = LOL_V1_ROOT / "high"

    low_images = sorted(
        low_dir.glob("*")
    )

    if LOL_V1_LIMIT is not None:
        low_images = low_images[:LOL_V1_LIMIT]

    psnr_values = []
    ssim_values = []

    print("\n" + "=" * 70)
    print("LOL-v1 EVALUATION")
    print("=" * 70)

    print(
        f"Images evaluated: {len(low_images)}"
    )

    for index, low_path in enumerate(
        low_images,
        start=1
    ):

        high_path = (
            high_dir / low_path.name
        )

        if not high_path.exists():
            print(
                f"WARNING: Missing reference "
                f"for {low_path.name}"
            )
            continue

        low_image = load_image(
            low_path
        )

        reference = load_image(
            high_path
        )

        with torch.inference_mode():

            enhanced = enhance_image(
                model,
                enhancement,
                refinement,
                low_image
            )

        enhanced_np = tensor_to_numpy(
            enhanced
        )

        reference_np = tensor_to_numpy(
            reference
        )

        psnr, ssim = calculate_metrics(
            enhanced_np,
            reference_np
        )

        psnr_values.append(psnr)
        ssim_values.append(ssim)

        print(
            f"[{index:02d}] "
            f"{low_path.name:15s} "
            f"PSNR: {psnr:7.3f} dB   "
            f"SSIM: {ssim:.4f}"
        )

    if psnr_values:

        print("\nLOL-v1 FULL-DATASET AVERAGE:")

        print(
            f"PSNR: "
            f"{sum(psnr_values) / len(psnr_values):.3f} dB"
        )

        print(
            f"SSIM: "
            f"{sum(ssim_values) / len(ssim_values):.4f}"
        )


# ============================================================
# LOL-v2 evaluation
# ============================================================

def evaluate_lol_v2(
    model,
    enhancement,
    refinement
):

    low_dir = LOL_V2_ROOT / "Low"
    normal_dir = LOL_V2_ROOT / "Normal"

    low_images = sorted(
        low_dir.glob("*")
    )

    if LOL_V2_LIMIT is not None:
        low_images = low_images[:LOL_V2_LIMIT]

    psnr_values = []
    ssim_values = []

    print("\n" + "=" * 70)
    print("LOL-v2 REAL TEST EVALUATION")
    print("=" * 70)

    print(
        f"Images evaluated: {len(low_images)}"
    )

    for index, low_path in enumerate(
        low_images,
        start=1
    ):

        # low00690.png → normal00690.png
        normal_name = low_path.name.replace(
            "low",
            "normal",
            1
        )

        normal_path = (
            normal_dir / normal_name
        )

        if not normal_path.exists():
            print(
                f"WARNING: Missing reference "
                f"for {low_path.name}"
            )
            continue

        low_image = load_image(
            low_path
        )

        reference = load_image(
            normal_path
        )

        with torch.inference_mode():

            enhanced = enhance_image(
                model,
                enhancement,
                refinement,
                low_image
            )

        enhanced_np = tensor_to_numpy(
            enhanced
        )

        reference_np = tensor_to_numpy(
            reference
        )

        psnr, ssim = calculate_metrics(
            enhanced_np,
            reference_np
        )

        psnr_values.append(psnr)
        ssim_values.append(ssim)

        print(
            f"[{index:03d}] "
            f"{low_path.name:15s} "
            f"PSNR: {psnr:7.3f} dB   "
            f"SSIM: {ssim:.4f}"
        )

    if psnr_values:

        print("\nLOL-v2 FULL-DATASET AVERAGE:")

        print(
            f"PSNR: "
            f"{sum(psnr_values) / len(psnr_values):.3f} dB"
        )

        print(
            f"SSIM: "
            f"{sum(ssim_values) / len(ssim_values):.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("LOW-LIGHT ENHANCEMENT FULL DATASET EVALUATION")
    print("=" * 70)

    print(
        f"\nCheckpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print(
        f"\nDevice:"
    )

    print(
        DEVICE
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    enhancement = (
        AdaptiveEnhancement()
        .to(DEVICE)
    )

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

    evaluate_lol_v1(
        model,
        enhancement,
        refinement
    )

    # --------------------------------------------------------
    # LOL-v2
    # --------------------------------------------------------

    evaluate_lol_v2(
        model,
        enhancement,
        refinement
    )

    # --------------------------------------------------------
    # ExDark
    # --------------------------------------------------------

    evaluate_exdark_no_reference(
        model,
        enhancement,
        refinement
    )

    print("\n" + "=" * 70)
    print("FULL DATASET EVALUATION COMPLETED")
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()