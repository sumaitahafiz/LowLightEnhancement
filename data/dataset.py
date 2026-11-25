from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class ExDarkDataset(Dataset):
    """
    PyTorch Dataset for the ExDark low-light image dataset.

    ExDark contains unpaired low-light images organized
    into object-category folders.
    """

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    def __init__(self, root_dir, image_size=256):
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset folder does not exist:\n{self.root_dir}"
            )

        self.image_paths = self._collect_images()

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images were found in:\n{self.root_dir}"
            )

    def _collect_images(self):
        """
        Recursively find all supported image files.
        """

        image_paths = []

        for file_path in self.root_dir.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.IMAGE_EXTENSIONS
            ):
                image_paths.append(file_path)

        return sorted(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        # Read image using OpenCV.
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Could not read image:\n{image_path}"
            )

        # OpenCV reads BGR.
        # Convert BGR → RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to a fixed size.
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        # Convert uint8 [0,255] → float32 [0,1].
        image = image.astype("float32") / 255.0

        # HWC → CHW.
        image = torch.from_numpy(image).permute(2, 0, 1)

        return {
            "image": image,
            "path": str(image_path),
        }