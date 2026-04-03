import torch
from skimage import io
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose
from torchvision.tv_tensors import BoundingBoxes
from pathlib import Path
from src.data.setup import SetupNeonTreeData
from src.const import LOGGER, DEVICE
from typing import Literal


class TreeImageDataset(Dataset):
    # Validation and training split, shared across all instances of TreeImageDataset
    train_indices = []
    val_indices = []

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        transforms: list | None = None,
    ):
        """
        Tree Image Dataset. Expects a directory with an `images/` subdirectory of .tif files
        and a `boxes/` subdirectory of .npy files sharing the same stems.
        Each .tif is a (4, 400, 400) float32 tensor (RGB + CHM, normalised to [0, 1]).
        Each .npy contains an (N, 4) float32 array of bounding boxes in XYXY pixel format.
        """
        self.dirs = (
            SetupNeonTreeData()
        )  # Initialize dirs to ensure data is set up and paths are available

        # Collect all image paths for the selected split and set input directory
        if split == "train" or split == "val":
            self.input_dir = self.dirs.train
        elif split == "test":
            self.input_dir = self.dirs.test

        self.paths = sorted(
            (self.input_dir / "images").glob("*.tif")
        )  # list of image paths, sorted for consistency

        # For train/val splits, filter paths based on indices
        if split == "train":
            if not self.train_indices:
                self._create_val_split()  # Create val split and populate indices
            self.paths = [self.paths[i] for i in self.train_indices]
        elif split == "val":
            if not self.val_indices:
                self._create_val_split()  # Create val split and populate indices
            self.paths = [self.paths[i] for i in self.val_indices]

        # Set up transforms if provided
        if transforms:
            self.transform = Compose(transforms)
        else:
            self.transform = None

        # Set device and loading strategy based on available VRAM
        self.device = DEVICE

        vram = 0
        # Check available VRAM to set eager/lazy loading
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
            if vram >= 16:  # Threshold for eager loading
                self.loading_is_eager = True
            else:
                self.loading_is_eager = False
        else:
            self.loading_is_eager = False  # Default to lazy loading if no GPU

        LOGGER.info(
            f"VRAM: {vram:.2f} GB, Loading strategy: {'Eager' if self.loading_is_eager else 'Lazy'}"
        )

    def _create_val_split(self, val_ratio: float = 0.2):
        """
        Creates a validation split from the training data. Randomly selects a percentage of
        the training indices to be used for validation and stores them in class variables.
        This ensures that all instances of TreeImageDataset share the same train/val split.
        Only called if the split hasn't been created before to avoid overwriting existing splits.
        """
        total_train_samples = len(list((self.dirs.train / "images").glob("*.tif")))
        indices = list(range(total_train_samples))
        np.random.shuffle(indices)

        val_size = int(total_train_samples * val_ratio)
        self.val_indices = indices[:val_size]
        self.train_indices = indices[val_size:]
        LOGGER.info(
            f"Created validation split: {len(self.train_indices)} train samples, {len(self.val_indices)} val samples."
        )

    def __len__(self) -> int:
        """
        Returns the number of image/box pairs in the dataset.
        """
        return len(self.paths)

    def data(self) -> list[tuple[Tensor, BoundingBoxes]]:
        """
        Loads all data into memory if eager loading is enabled.
        Returns a list of (image, boxes) tuples.
        """
        if not hasattr(self, "_data"):
            self._data = []
            for img_path in self.paths:
                self.data.append(self._load_data_point(img_path))
        return self._data

    def _load_data_point(self, path: Path) -> tuple[Tensor, BoundingBoxes]:
        """
        Loads a single image and its corresponding bounding boxes from disk.
        Returns the image tensor and BoundingBoxes object.
        """
        img_path = path
        box_path = self.input_dir / "boxes" / (img_path.stem + ".npy")

        image = torch.from_numpy(io.imread(img_path)).to(
            device=self.device
        )  # Shape: (400, 400, 4)
        boxes = BoundingBoxes(
            np.load(box_path), format="XYXY", canvas_size=(400, 400)
        ).to(
            device=self.device
        )  # Shape: (N, 4)

        image = image.permute(2, 0, 1)  # Convert to (4, 400, 400)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        target = {
            "boxes": boxes,
            "labels": torch.ones(len(boxes), dtype=torch.int64, device=self.device),
        }  # Dummy labels (all ones)

        return image, target

    def __getitem__(self, idx: int) -> tuple[Tensor, BoundingBoxes]:
        """
        Returns the image tensor and bounding boxes for the sample at `idx`.
        Data is loaded eagerly or lazily based on the VRAM availability.
        Image shape: (4, 400, 400), float32.
        Boxes: BoundingBoxes of shape (N, 4) in XYXY format.
        If transforms are set, they are applied jointly to both image and boxes.
        """

        if self.loading_is_eager:
            return self.data[idx]
        else:
            return self._load_data_point(self.paths[idx])
