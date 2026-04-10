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


class GeneralizedBoxIoULoss(
    nn.Module
):  # Useless wrapper around generalized_box_iou to make it compatible with Trainer's expected criterion interface
    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super(GeneralizedBoxIoULoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        # Compute the Generalized IoU loss
        iou_loss = 1 - generalized_box_iou(
            pred_boxes, target_boxes, reduction=self.reduction, eps=self.eps
        )
        return iou_loss.mean()
