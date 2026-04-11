import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou


def detection_collate_fn(batch):
    """Custom collate function for object detection datasets."""
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)

    # Stack images normally: (B, C, H, W)
    images = torch.stack(images, dim=0)
    return images, targets
