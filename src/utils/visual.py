from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io
from torchvision.tv_tensors import BoundingBoxes
from torchvision.utils import draw_bounding_boxes
import torch
import numpy as np

from src.const import PT_DATA_PATH, LOGGER


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
    plt.show()
