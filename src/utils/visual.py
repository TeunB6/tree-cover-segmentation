from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io
from torchvision.tv_tensors import BoundingBoxes
from torchvision.utils import draw_bounding_boxes

from rich.table import Table

import torch
import numpy as np

from src.const import PT_DATA_PATH, LOGGER

def print_table(data: dict, title: str = "Table") -> None:
    table = Table(title=title)
    for key in data.keys():
        table.add_column(key, justify="center", header_style="cyan", )
    table.add_row(*[str(value.item() if isinstance(value, torch.Tensor) else value) for value in data.values()])
    LOGGER.log_and_print(table)

def view_image_with_boxes_from_name(name: str, split: str = "train") -> None:
    bb_path = PT_DATA_PATH / split / "boxes" / f"{name}.npy"
    img_path = PT_DATA_PATH / split / "images" / f"{name}.tif"
    boxes = torch.from_numpy(np.load(bb_path))
    image = torch.from_numpy(io.imread(img_path))

    bounding_boxes = BoundingBoxes(boxes, format="XYXY", canvas_size=image.shape[:2])

    view_image_with_boxes(image, bounding_boxes)


def view_image_with_boxes(
    image: torch.Tensor,
    boxes: BoundingBoxes,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    LOGGER.debug(f"Plotting... Image shape: {image.shape}, Boxes shape: {boxes.shape}")
    # Check CHW vs HWC format and convert to HWC if needed
    if (
        image.shape[0] == 4
    ):  # CHW format TODO: Will break if we add more than 4 channels
        image = image.permute(1, 2, 0)  # Convert to HWC for plotting

    # Separate RGB and CHM channels
    rgb_image = image[:, :, :3]  # First 3 channels are RGB
    chm_image = image[:, :, 3]  # 4th channel is CH

    # Draw bounding boxes on the RGB image
    image_with_boxes = draw_bounding_boxes(
        rgb_image.permute(2, 0, 1),  # Convert to CxHxW format
        boxes=boxes,
        colors="red",
        width=2,
    )

    # Create plot
    fig, axes = plt.subplots(1, 2 if image.shape[2] == 4 else 1, figsize=(12, 6))
    axes[0].imshow(
        image_with_boxes.cpu().permute(1, 2, 0)
    )  # Convert back to HxWxC format
    axes[0].set_title("RGB Image with Bounding Boxes")
    axes[0].axis("off")
    if image.shape[2] == 4:
        axes[1].imshow(chm_image.cpu(), cmap="viridis")
        axes[1].set_title("CHM Channel")
        axes[1].axis("off")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        LOGGER.info(f"Saved image with boxes to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def view_prediction(image: torch.Tensor, pred_boxes: BoundingBoxes, target_boxes: BoundingBoxes, save_path: Path | None = None, show: bool = True) -> None:
    # Draw predicted boxes in red and ground truth boxes in green
    if image.shape[0] == 4: 
        rgb_image = image[:3, :, :] # Use RGB channels and convert to HWC for plotting
        chm_image = image[3, :, :] # CHM channel
    
    rgb_image_with_boxes = draw_bounding_boxes(
        rgb_image,
        boxes=pred_boxes,
        colors="red",
        width=2,
    )
    rgb_image_with_boxes = draw_bounding_boxes(
        rgb_image_with_boxes,
        boxes=target_boxes,
        colors="green",
        width=2,
    )
    
    
    # Create plot with RGB and CHM side by side
    fig, axes = plt.subplots(1, 2 if image.shape[0] == 4 else 1, figsize=(12, 6))
    axes[0].imshow(rgb_image_with_boxes.cpu().permute(1, 2, 0))  # Convert back to HxWxC format
    axes[0].set_title("Predicted Boxes (Red) vs Ground Truth Boxes (Green)")
    axes[0].axis("off")
    
    if image.shape[0] == 4:
        axes[1].imshow(chm_image.cpu(), cmap="viridis")
        axes[1].set_title("CHM Channel")
        axes[1].axis("off")
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        LOGGER.info(f"Saved image with boxes to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def view_image(
    image: torch.Tensor,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    # Check CHW vs HWC format and convert to HWC if needed
    if (
        image.shape[0] == 4
    ):  # CHW format TODO: Will break if we add more than 4 channels
        image = image.permute(1, 2, 0)  # Convert to HWC for plotting

    rgb_image = image[:, :, :3]  # First 3 channels are RGB
    chm_image = image[:, :, 3]  # 4th channel is CH

    # Create plot
    fig, axes = plt.subplots(1, 2 if image.shape[2] == 4 else 1, figsize=(12, 6))
    axes[0].imshow(rgb_image.cpu())  # Convert back to HxWxC format
    axes[0].set_title("RGB Image with Bounding Boxes")
    axes[0].axis("off")
    if image.shape[2] == 4:
        axes[1].imshow(chm_image.cpu(), cmap="viridis")
        axes[1].set_title("CHM Channel")
        axes[1].axis("off")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        LOGGER.info(f"Saved image with boxes to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
