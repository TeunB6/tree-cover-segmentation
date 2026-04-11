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
from rich.progress import track
from torchvision.transforms.v2 import functional as F


class TreeImageDataset(Dataset):
    # Validation and training split, shared across all instances of TreeImageDataset
    train_indices = []
    val_indices = []

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        transforms: list | None = None,
        force_lazy_loading: bool = False,
        transform_inflate_factor: int = 1,
    ):
        """
        Tree Image Dataset. Expects a directory with an `images/` subdirectory of .tif files
        and a `boxes/` subdirectory of .npy files sharing the same stems.
        Each .tif is a (4, 400, 400) float32 tensor (RGB + CHM, normalised to [0, 1]).
        Each .npy contains an (N, 4) float32 array of bounding boxes in XYXY pixel format.
        
        Args:
            split (str): One of "train", "val", or "test" to specify the dataset split.
            transforms (list, optional): A list of torchvision transforms to apply jointly to images and boxes. Defaults to None.
            force_lazy_loading (bool, optional): If True, forces lazy loading regardless of VRAM. Defaults to False.
            transform_inflate_factor (int, optional): Factor by which to inflate the dataset size through transforms. Only used if transforms are provided and if loading is eager. Defaults to 1 (no inflation).
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
            self.transform = (
                Compose(transforms)
                if not isinstance(transforms, Compose)
                else transforms
            )
        else:
            self.transform = None

        # Set device and loading strategy based on available VRAM
        self.device = DEVICE
        self.transform_inflate_factor = transform_inflate_factor

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

        if force_lazy_loading:
            self.loading_is_eager = False

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

    @property
    def data(self) -> list[tuple[Tensor, BoundingBoxes]]:
        """
        Loads all data into memory if eager loading is enabled.
        Returns a list of (image, boxes) tuples.
        """
        if not hasattr(self, "_data"):
            self._data = []
            for img_path in track(
                self.paths, description="Loading data:", total=len(self.paths)
            ):
                for _ in range(self.transform_inflate_factor):  # Inflate dataset if factor > 1
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
        )  # Shape: (400, 400, 4) (in most cases :( )
        
        image = image.permute(2, 0, 1)  # Convert to (4, 400, 400)

        # Check shape and pad to (N, 400, 400) if necessary #TODO: Move remaining data validation and cleaning done here to setup 
        if image.shape[1] != 400 or image.shape[2] != 400:
            LOGGER.warning(
                f"Image {img_path.stem} has unexpected shape {image.shape}. Resizing to (400, 400)."
            )
            image = F.resize(image, (400, 400))  # Ensure image is 400x400

        
        # Handle NaN values in the image (replace with 0)
        if torch.isnan(image).any():
            LOGGER.warning(
                f"Image {img_path.stem} contains NaN values. Replacing with 0."
            )
            image = torch.where(
                torch.isnan(image), torch.tensor(0.0, device=self.device), image
            )

        boxes = BoundingBoxes(
            np.load(box_path), format="XYXY", canvas_size=(400, 400)
        ).to(
            device=self.device
        )  # Shape: (N, 4)


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
        
    def get_site_name(self, idx: int) -> str:
        """
        Utility method to get the site name (stem of the filename) for a given index.
        """
        return self.paths[idx].stem
