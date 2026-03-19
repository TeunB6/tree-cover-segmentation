"""
Compute channel-wise mean and stdev and CHM max for normalization transform
"""

from pathlib import Path
import numpy as np
import torch
from typing import Iterable
from rich.progress import track


def _collect_files(data_dir: Path | Iterable[Path]) -> list[Path]:
    files = []
    if isinstance(data_dir, Path):
        files.extend(sorted(data_dir.glob("*.pt")))
    elif isinstance(data_dir, Iterable):
        for dir in data_dir:
            files.extend(sorted(dir.glob("*.pt")))
    return files


def get_images(data_dir: Path | Iterable[Path]):
    files = _collect_files(data_dir)
    unpickles = [
        torch.load(file, weights_only=False)
        for file in track(files, description="Loading images...")
    ]
    images = [np.asarray(image) for image, boxes in unpickles]
    for img in images:
        img[3] = np.clip(img[3], 0, None)
    images = np.concatenate(
        [np.nan_to_num(img).reshape(img.shape[0], -1) for img in images], axis=1
    )
    return images


def chm_max(data_dir: Path | Iterable[Path]):
    files = _collect_files(data_dir)
    max_val = 0.0
    for file in track(files, description="Computing CHM max..."):
        image, _ = torch.load(file, weights_only=False)
        chm = np.asarray(image[3])
        max_val = max(max_val, float(np.nanmax(np.clip(chm, 0, None))))
    return max_val


def means(images):
    return np.mean(images, axis=1)


def stdevs(images):
    return np.std(images, axis=1)


if __name__ == "__main__":
    base_path = Path("data/pt_data")
    splits = [base_path / "train", base_path / "test"]
    print("CHM max:             ", chm_max(splits))
    images = get_images(splits)
    print("Means:               ", means(images))
    print("Standard Deviations: ", stdevs(images))
