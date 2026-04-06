import random

import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as F


# ─────────────────────────────────────────────
#  Original transform (unchanged)
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
#  1. Flip and Rotate
# ─────────────────────────────────────────────

class RandomFlipAndRotate(torch.nn.Module):
    """
    Randomly flips (horizontal and/or vertical) and rotates the image.

    Satellite/aerial imagery has no canonical orientation — a forest looks
    the same from any angle — so aggressive flipping and rotation are safe
    and effective augmentations.

    Args:
        p_hflip  (float): probability of horizontal flip.       Default 0.5
        p_vflip  (float): probability of vertical flip.         Default 0.5
        degrees  (float): max rotation angle in degrees (±).    Default 90
    """

    def __init__(self, p_hflip=0.5, p_vflip=0.5, degrees=90):
        super().__init__()
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.degrees = degrees

    def forward(self, img, label=None):
        # Random horizontal flip
        if random.random() < self.p_hflip:
            img = F.horizontal_flip(img)

        # Random vertical flip
        if random.random() < self.p_vflip:
            img = F.vertical_flip(img)

        # Random rotation between -degrees and +degrees
        angle = random.uniform(-self.degrees, self.degrees)
        img = F.rotate(img, angle)

        if label is not None:
            return img, label
        return img


# ─────────────────────────────────────────────
#  2. Translations and Shear
# ─────────────────────────────────────────────

class RandomTranslationShear(torch.nn.Module):
    """
    Randomly applies translation and shear to the image.

    Translation shifts the image spatially; shear skews it. Both simulate
    slight variations in sensor angle and flight path for aerial imagery.

    Args:
        translate (tuple): max horizontal and vertical shift as a fraction
                           of image size. E.g. (0.1, 0.1) = up to 10%.
                           Default (0.1, 0.1)
        shear     (float): max shear angle in degrees. Default 10
    """

    def __init__(self, translate=(0.1, 0.1), shear=10):
        super().__init__()
        self.translate = translate
        self.shear = shear

    def forward(self, img, label=None):
        # v2.RandomAffine handles both translation and shear cleanly
        transform = v2.RandomAffine(
            degrees=0,                  # no rotation here (handled separately)
            translate=self.translate,
            shear=self.shear,
        )
        img = transform(img)

        if label is not None:
            return img, label
        return img


# ─────────────────────────────────────────────
#  3. Gaussian Blur (mimic sensor noise)
# ─────────────────────────────────────────────

class RandomGaussianBlur(torch.nn.Module):
    """
    Randomly applies Gaussian blur to simulate sensor noise and atmospheric
    distortion common in satellite and aerial imagery.

    Args:
        kernel_size (int):   size of the blur kernel (must be odd). Default 5
        sigma       (tuple): range of blur strength (min, max).
                             Default (0.1, 2.0)
        p           (float): probability of applying the blur. Default 0.5
    """

    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.5):
        super().__init__()
        self.p = p
        self.blur = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, img, label=None):
        if random.random() < self.p:
            img = self.blur(img)

        if label is not None:
            return img, label
        return img


# ─────────────────────────────────────────────
#  4. Sat-CutMix
# ─────────────────────────────────────────────

class SatCutMix(torch.nn.Module):
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


# ─────────────────────────────────────────────
#  5. Sat-SlideMix
# ─────────────────────────────────────────────

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

    def __init__(self, max_shift=0.5, p=0.5, direction="both"):
        super().__init__()
        assert direction in ("horizontal", "vertical", "both"), \
            "direction must be 'horizontal', 'vertical', or 'both'"
        self.max_shift = max_shift
        self.p = p
        self.direction = direction

    def forward(self, img, label=None):
        """
        Args:
            img   (Tensor): single image [C, H, W]
            label (optional): passed through unchanged
        """
        if random.random() > self.p:
            if label is not None:
                return img, label
            return img

        assert img.dim() == 3, "SatSlideMix expects a single image: [C, H, W]"
        _, H, W = img.shape

        if self.direction in ("horizontal", "both"):
            shift_w = random.randint(1, int(W * self.max_shift))
            img = torch.roll(img, shifts=shift_w, dims=2)  # dim 2 = width

        if self.direction in ("vertical", "both"):
            shift_h = random.randint(1, int(H * self.max_shift))
            img = torch.roll(img, shifts=shift_h, dims=1)  # dim 1 = height

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
    return v2.Compose([
        ToTensor(),                                      # convert to float32 tensor
        RandomFlipAndRotate(p_hflip=0.5, p_vflip=0.5, degrees=90),
        RandomTranslationShear(translate=(0.1, 0.1), shear=10),
        RandomGaussianBlur(kernel_size=5, sigma=(0.1, 2.0), p=0.5),
        SatSlideMix(max_shift=0.5, p=0.5, direction="both"),
    ])


def get_val_transforms():
    """
    Returns the validation/inference pipeline — no augmentation, just
    converts to tensor. Keeps validation deterministic and comparable.
    """
    return v2.Compose([
        ToTensor(),
    ])
