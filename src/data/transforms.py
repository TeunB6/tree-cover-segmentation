import random

import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as F
import numpy as np
from torchvision.tv_tensors import BoundingBoxes

from src.utils.visual import view_image_with_boxes
from src.const import LOGGER
from pathlib import Path


class ToTensor(torch.nn.Module):
    """
    Equivalent to transforms v1 ToTensor.
    Converts a PIL image or numpy array to a float32 tensor scaled to [0, 1].
    Works with or without a label.
    """

    def forward(self, img, label=None):
        if label is not None:
            return (
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(img),
                label,
            )
        else:
            return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(
                img
            )


class SatCutMix(
    torch.nn.Module
):  # TODO: take into account the bounding boxes when cutting and pasting
    """
    Satellite CutMix: cuts a random rectangular patch from a second image
    and pastes it onto the first image.

    Operates on a BATCH of images [N, C, H, W] — pairs within the batch
    are mixed with each other (img[0] with img[1], img[2] with img[3], etc.).

    This teaches the model to detect trees regardless of their local
    surroundings, improving robustness to varying forest patch appearances.

    Args:
        alpha (float): controls the size of the cut region via a Beta
                       distribution. Higher alpha = larger cuts. Default 1.0
        p     (float): probability of applying CutMix to each pair. Default 0.5

    Input shape:  [N, C, H, W]  — must be a batch (N >= 2)
    Output shape: [N, C, H, W]  — same shape, pairs are mixed
    """

    def __init__(self, alpha=1.0, p=0.5):
        super().__init__()
        self.alpha = alpha
        self.p = p

    def _sample_box(self, lam, H, W):
        """Sample a random bounding box given mix ratio lam."""
        cut_ratio = (1.0 - lam) ** 0.5
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cx = random.randint(0, W)
        cy = random.randint(0, H)

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)

        return x1, y1, x2, y2

    def forward(self, imgs, labels=None):
        """
        Args:
            imgs   (Tensor): batch of images [N, C, H, W]
            labels (Tensor, optional): batch of labels [N, ...]
        """
        assert imgs.dim() == 4, "SatCutMix expects a batch: [N, C, H, W]"
        N, C, H, W = imgs.shape
        assert N >= 2, "Batch size must be at least 2 for CutMix"

        result = imgs.clone()

        # Process pairs: (0,1), (2,3), ...
        for i in range(0, N - 1, 2):
            if random.random() > self.p:
                continue

            # Sample mix ratio from Beta distribution
            lam = random.betavariate(self.alpha, self.alpha)
            x1, y1, x2, y2 = self._sample_box(lam, H, W)

            # Paste patch from image i+1 onto image i, and vice versa
            result[i, :, y1:y2, x1:x2] = imgs[i + 1, :, y1:y2, x1:x2]
            result[i + 1, :, y1:y2, x1:x2] = imgs[i, :, y1:y2, x1:x2]

        if labels is not None:
            return result, labels
        return result


class SatSlideMix(torch.nn.Module):
    """
    Satellite SlideMix: rolls (slides) the image along one or both axes so
    that the part that falls off one edge reappears on the opposite edge.

    Works on a SINGLE image [C, H, W] — no second image needed.

    This is valid for aerial/satellite imagery because the texture is
    spatially uniform — a rolled forest patch still looks like a realistic
    forest patch. It forces the model to be invariant to the absolute
    position of trees within a patch.

    Args:
        max_shift (float): maximum roll distance as a fraction of image
                           size. E.g. 0.5 = roll up to 50%. Default 0.5
        p         (float): probability of applying the roll. Default 0.5
        direction (str):   axis to roll along — 'horizontal', 'vertical',
                           or 'both'. Default 'both'

    Input shape:  [C, H, W]  — single image
    Output shape: [C, H, W]  — same shape, content is rolled
    """

    def __init__(self, max_shift=0.5, direction="both"):
        super().__init__()
        assert direction in (
            "horizontal",
            "vertical",
            "both",
            "random",
        ), "direction must be 'horizontal', 'vertical', or 'both'"
        self.max_shift = max_shift
        self.direction = (
            direction
            if direction != "random"
            else random.choice(["horizontal", "vertical"])
        )

    def forward(self, img, label: torch.Tensor | None = None):
        """
        Args:
            img   (Tensor): single image [C, H, W]
            label (optional): passed through unchanged
        """
        assert img.dim() == 3, "SatSlideMix expects a single image: [C, H, W]"
        _, H, W = img.shape

        LOGGER.debug(
            f"SatSlideMix: label before roll: {label if label is not None and len(label) > 0 else 'N/A'}, type: {type(label) if label is not None else 'N/A'}"
        )

        if self.direction in ("horizontal", "both"):
            shift_w = random.randint(1, int(W * self.max_shift))
            img = torch.roll(img, shifts=shift_w, dims=2)  # dim 2 = width
            label = self._wrap_bounding_boxes(
                label, shift_w, W, indexes=[0, 2]
            )  # adjust x-coords

        if self.direction in ("vertical", "both"):
            shift_h = random.randint(1, int(H * self.max_shift))
            img = torch.roll(img, shifts=shift_h, dims=1)  # dim 1 = height
            if label is not None:
                label[:, [1, 3]] = (label[:, [1, 3]] + shift_h) % H  # adjust y-coords
                label = self._wrap_bounding_boxes(
                    label, shift_h, H, indexes=[1, 3]
                )  # adjust y-coords

        if label is not None:
            LOGGER.debug(
                f"SatSlideMix: label after roll: {label if label is not None  and len(label) > 0 else 'N/A'}, type: {type(label) if label is not None else 'N/A'}"
            )
            return img, label
        return img

    def _wrap_bounding_boxes(
        self, boxes: torch.Tensor | None, shift: int, dim_size: int, indexes: list[int]
    ) -> torch.Tensor | None:
        """Helper function to wrap bounding boxes when rolling. A box is split into two if it goes out of bounds."""
        if boxes is None or len(boxes) == 0:
            return boxes
        boxes_is_tv_tensor = isinstance(boxes, BoundingBoxes)
        shifted_boxes = boxes.clone()
        shifted_boxes[:, indexes] = (shifted_boxes[:, indexes] + shift) % dim_size

        # Identify boxes that were wrapped around and create wrapped versions
        wrong_wrapped = shifted_boxes[:, indexes[0]] > shifted_boxes[:, indexes[1]]
        wrapped = shifted_boxes[wrong_wrapped].clone()

        # trim out of bounds part from original boxes
        shifted_boxes[wrong_wrapped, indexes[1]] = dim_size

        # create wrapped part as new boxes
        wrapped[:, indexes[0]] = 0

        merged = torch.cat([shifted_boxes, wrapped], dim=0)

        # Preserve torchvision BoundingBoxes metadata for downstream transforms.
        if boxes_is_tv_tensor:
            return BoundingBoxes(
                merged,
                format=boxes.format,
                canvas_size=boxes.canvas_size,
                dtype=boxes.dtype,
                device=boxes.device,
            )

        return merged


class RandomCompose(torch.nn.Module):
    """
    Randomly applies one of the given transforms to the image.

    This can be used to add more variety by randomly choosing between
    different augmentation strategies.

    Args:
        transforms (list): list of transform modules to choose from.
    """

    def __init__(self, transforms: list, weights: list = None, debug: bool = False):
        super().__init__()
        self.transforms = transforms
        if weights is not None:
            assert len(weights) == len(
                transforms
            ), "Weights length must match transforms"
            if not np.isclose(sum(weights), 1.0):  # normalize weights to sum to 1
                weights = [w / sum(weights) for w in weights]
            self.weights = weights
        else:
            self.weights = [1.0 / len(transforms)] * len(transforms)

        self.debug = debug

    def forward(self, img, label=None):
        transform = np.random.choice(self.transforms, 1, p=self.weights)[0]
        out = transform(img, label)
        if self.debug:
            view_image_with_boxes(
                out[0],
                out[1],
                save_path=Path("out")
                / f"debug_random_compose_{transform.__class__.__name__}.png",
            )
        return out


class IdentityTransform(torch.nn.Module):
    """
    A no-op transform that returns the input unchanged.

    Useful as a placeholder or for testing the pipeline without augmentation.
    """

    def forward(self, img, label=None):
        if label is not None:
            return img, label
        return img


# ─────────────────────────────────────────────
#  Compose all transforms into a ready-to-use pipeline
# ─────────────────────────────────────────────


def get_train_transforms():
    """
    Returns the full single-image augmentation pipeline for training.

    Note: SatCutMix is NOT included here because it requires a batch.
    Apply SatCutMix separately after the DataLoader assembles batches.

    Usage in training loop:
        single_transforms = get_train_transforms()
        cutmix = SatCutMix(alpha=1.0, p=0.5)

        for imgs, labels in dataloader:
            imgs, labels = cutmix(imgs, labels)   # batch-level
            # imgs already had single-image transforms applied in the dataset
    """
    return v2.Compose(
        [
            ToTensor(),
            RandomCompose(
                [
                    # Rotations and flips
                    v2.RandomHorizontalFlip(p=1.0),  # always apply flip when chosen
                    v2.RandomVerticalFlip(p=1.0),
                    v2.RandomRotation(degrees=(-45, 45)),
                    # Shear and translation
                    v2.RandomAffine(translate=(0.1, 0.1), shear=10, degrees=0),
                    v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                    SatSlideMix(max_shift=0.5, direction="random"),
                    IdentityTransform(),
                ],
                weights=[
                    1, # rotate
                    1, # hflip
                    1, # vflip
                    3, # affine
                    3, # blur
                    3, # slidemix 
                    3, # identity
                ],
                debug=False,
            ),
            v2.SanitizeBoundingBoxes(labels_getter=lambda x: x[1]),
        ]
    )


def get_val_transforms():
    """
    Returns the validation/inference pipeline — no augmentation, just
    converts to tensor. Keeps validation deterministic and comparable.
    """
    return v2.Compose(
        [
            ToTensor(),
        ]
    )
