import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import (
    Normalize,
    Compose,
    RandomCrop,
    SanitizeBoundingBoxes,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from pathlib import Path
from src.const import CHANNEL_MEANS, CHANNEL_STDEVS
from src.data.setup import SetupNeonTreeData


class TreeImageDataset(Dataset):
    def __init__(
        self,
        input_dir: Path | str,
        transforms: list | None = None,
    ):
        """
        Tree Image Dataset. Expects a directory path filled with .pt files which each contain a
        two-tuple of a tensor of a 400x400xC image file and a tensor of the bounding boxes.
        """
        SetupNeonTreeData()
        self.input_dir = input_dir if type(input_dir) == Path else Path(input_dir)
        self.paths = sorted(self.input_dir.glob("*.pt"))
        if transforms:
            self.transform = Compose(transforms)
        else:
            self.transform = Compose(
                [
                    RandomCrop(400),
                    Normalize(CHANNEL_MEANS, CHANNEL_STDEVS),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    SanitizeBoundingBoxes(),
                ]
            )

    def __len__(self) -> int:
        """
        Returns the size of the dataset; that is, the number of pt files in the given directory.
        """
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Gets a specific entry in the dataset and returns the image and bounding boxes tensors.
        """
        file, boxes = torch.load(self.paths[idx])
        file = self.transform(file)
        return file, boxes
