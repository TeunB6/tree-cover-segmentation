import torch
import tifffile
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose
from torchvision.tv_tensors import BoundingBoxes
from pathlib import Path
from src.data.setup import SetupNeonTreeData


class TreeImageDataset(Dataset):
    def __init__(
        self,
        input_dir: Path | str,
        transforms: list | None = None,
    ):
        """
        Tree Image Dataset. Expects a directory with an `images/` subdirectory of .tif files
        and a `boxes/` subdirectory of .npy files sharing the same stems.
        Each .tif is a (4, 400, 400) float32 tensor (RGB + CHM, normalised to [0, 1]).
        Each .npy contains an (N, 4) float32 array of bounding boxes in XYXY pixel format.
        """
        SetupNeonTreeData()
        self.input_dir = input_dir if type(input_dir) == Path else Path(input_dir)
        self.paths = sorted((self.input_dir / "images").glob("*.tif"))
        if transforms:
            self.transform = Compose(transforms)
        else:
            self.transform = None

    def __len__(self) -> int:
        """
        Returns the number of image/box pairs in the dataset.
        """
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, BoundingBoxes]:
        """
        Returns the image tensor and bounding boxes for the sample at `idx`.
        Image shape: (4, 400, 400), float32.
        Boxes: BoundingBoxes of shape (N, 4) in XYXY format.
        If transforms are set, they are applied jointly to both image and boxes.
        """
        img_path = self.paths[idx]
        box_path = self.input_dir / "boxes" / (img_path.stem + ".npy")

        image = torch.from_numpy(tifffile.imread(img_path))
        boxes = BoundingBoxes(np.load(box_path), format="XYXY", canvas_size=(400, 400))

        if self.transform:
            image, boxes = self.transform(image, boxes)

        return image, boxes


if __name__ == "__main__":  # Example usage
    dataset = TreeImageDataset("data/pt_data/train")
    for i in range(5):
        img, boxes = dataset[i]
        print(f"Sample {i}: Image shape {img.shape}, Boxes shape {boxes.shape}")
