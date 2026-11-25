import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Dataset paths
# ============================================================

LOL_V1_ROOT = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\LOLdataset\eval15"
)

LOL_V2_ROOT = Path(
    r"D:\12th Trimester\DIP\DIP Paper\Datasets\LOL-v2"
    r"\LOL-v2\Real_captured\Test"
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 256

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# Image loading
# ============================================================

def load_image(path):

    image = Image.open(path).convert("RGB")

    return transform(image)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(image, reference):

    image = image.permute(1, 2, 0).numpy()
    reference = reference.permute(1, 2, 0).numpy()

    psnr = peak_signal_noise_ratio(
        reference,
        image,
        data_range=1.0
    )

    ssim = structural_similarity(
        reference,
        image,
        channel_axis=2,
        data_range=1.0
    )

    return psnr, ssim


# ============================================================
# LOL-v1
# ============================================================

def evaluate_lol_v1():

    low_dir = LOL_V1_ROOT / "low"
    high_dir = LOL_V1_ROOT / "high"

    low_images = sorted(low_dir.glob("*"))

    psnr_values = []
    ssim_values = []

    print("\n" + "=" * 70)
    print("LOL-v1 LOW-LIGHT BASELINE")
    print("=" * 70)

    for low_path in low_images:

        high_path = high_dir / low_path.name

        low = load_image(low_path)
        reference = load_image(high_path)

        psnr, ssim = calculate_metrics(
            low,
            reference
        )

        psnr_values.append(psnr)
        ssim_values.append(ssim)

        print(
            f"{low_path.name:15s} "
            f"PSNR: {psnr:7.3f} dB   "
            f"SSIM: {ssim:.4f}"
        )

    average_psnr = sum(psnr_values) / len(psnr_values)
    average_ssim = sum(ssim_values) / len(ssim_values)

    print("\nLOL-v1 baseline average:")
    print(f"PSNR: {average_psnr:.3f} dB")
    print(f"SSIM: {average_ssim:.4f}")


# ============================================================
# LOL-v2
# ============================================================

def evaluate_lol_v2():

    low_dir = LOL_V2_ROOT / "Low"
    normal_dir = LOL_V2_ROOT / "Normal"

    low_images = sorted(low_dir.glob("*"))

    psnr_values = []
    ssim_values = []

    print("\n" + "=" * 70)
    print("LOL-v2 REAL TEST LOW-LIGHT BASELINE")
    print("=" * 70)

    for low_path in low_images:

        normal_name = low_path.name.replace(
            "low",
            "normal",
            1
        )

        normal_path = normal_dir / normal_name

        low = load_image(low_path)
        reference = load_image(normal_path)

        psnr, ssim = calculate_metrics(
            low,
            reference
        )

        psnr_values.append(psnr)
        ssim_values.append(ssim)

        print(
            f"{low_path.name:15s} "
            f"PSNR: {psnr:7.3f} dB   "
            f"SSIM: {ssim:.4f}"
        )

    average_psnr = sum(psnr_values) / len(psnr_values)
    average_ssim = sum(ssim_values) / len(ssim_values)

    print("\nLOL-v2 baseline average:")
    print(f"PSNR: {average_psnr:.3f} dB")
    print(f"SSIM: {average_ssim:.4f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LOW-LIGHT INPUT BASELINE EVALUATION")
    print("=" * 70)

    evaluate_lol_v1()
    evaluate_lol_v2()

    print("\n" + "=" * 70)
    print("BASELINE EVALUATION COMPLETED")
    print("=" * 70)