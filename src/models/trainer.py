from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch
from rich.progress import track
import copy
from src.const import LOGGER, DEVICE
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable
from pathlib import Path
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou, generalized_box_iou

from src.models.faster_rcnn import FasterRCNNWrapper
from src.utils.misc import detection_collate_fn

from src.utils.visual import print_table

# Define plotting style.
plt.style.use("seaborn-v0_8-dark-palette")
plt.rcParams.update(
    {
        "figure.figsize": (12, 6),
        "axes.labelsize": 16,
        "axes.grid": True,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 18,
        "legend.fontsize": 16,
        "lines.linewidth": 2,
        "text.usetex": False,
        "font.family": "serif",
        "image.cmap": "magma",
    }
)


def train_faster_rcnn(
    wrapper: FasterRCNNWrapper,
    train_data: Dataset,
    val_data: Dataset,
    num_epochs: int = 10,
    optimizer: Callable = optim.AdamW,
    early_stopping: bool = True,
    patience: int = 3,
) -> dict:
    """Train a Faster R-CNN model.

    Args:
        model (FasterRCNNWrapper): The Faster R-CNN model to train.
        train_data (Dataset): The training dataset.
        val_data (Dataset): The validation dataset.
        test_data (Dataset): The test dataset.
        num_epochs (int, optional): The number of epochs to train for. Defaults to 10.
        optimizer (Callable, optional): The optimizer to use. Defaults to optim.AdamW.
        early_stopping (bool, optional): Whether to use early stopping. Defaults to True.
        patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 3.

    Returns:
        dict: A dictionary containing the training history and the best model state.
    """
    # Move model to device
    model = wrapper.model
    model.to(DEVICE)
    LOGGER.info(f"Start training: Model moved to {DEVICE} for training.")

    # Create data loaders
    train_loader = DataLoader(
        train_data, batch_size=16, shuffle=True, collate_fn=detection_collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size=16, shuffle=False, collate_fn=detection_collate_fn
    )

    optim = optimizer(model.parameters(), lr=1e-4)

    history = {"train_loss": [], "val_iou": []}
    best_model_state = copy.deepcopy(model.state_dict())
    best_val_map = 0.0
    epochs_no_improve = 0
    nan_batch_count = 0

    # Initialize MeanAveragePrecision metric
    map_metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", backend="faster_coco_eval"
    )
    map_metric = map_metric.to(DEVICE)

    for epoch in track(range(num_epochs), description="Epochs:", total=num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for batch_idx, (images, targets) in track(
            enumerate(train_loader), description="Batches:", total=len(train_loader)
        ):
            optim.zero_grad()
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                LOGGER.error(
                    f"Batch {batch_idx}: NaN/Inf in total loss: {losses}. Loss dict: {loss_dict}"
                )
                nan_batch_count += 1
                optim.zero_grad()
                continue

            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            total_train_loss += losses.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        LOGGER.info(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, NaN batches: {nan_batch_count}"
        )

        # Evaluation phase with mAP and IoU metrics
        model.eval()  # NOTE maybe we shouldn't do this, as we can not evaluate validation set loss to check over fitting. Disabling has big leakage risks due to batch norm and dropout
        map_metric.reset()
        total_iou = 0.0
        total_iou_targets = 0

        with torch.no_grad():
            for images, targets in val_loader:
                predictions = model(images)

                map_metric.update(predictions, targets)
                for prediction, target in zip(predictions, targets):
                    target_boxes = target["boxes"]
                    total_iou_targets += target_boxes.shape[0]

                    prediction_boxes = prediction["boxes"]
                    if prediction_boxes.numel() == 0 or target_boxes.numel() == 0:
                        continue

                    iou_matrix = box_iou(prediction_boxes, target_boxes)
                    total_iou += iou_matrix.max(dim=0).values.sum().item()

        metrics_dict = map_metric.compute()
        metrics_dict["val_iou"] = (
            total_iou / total_iou_targets if total_iou_targets > 0 else 0.0
        )

        # Initialize history keys on first compute, then append values
        for key in metrics_dict.keys():
            if key not in history:
                history[key] = []
            metric_value = (
                metrics_dict[key].item()
                if isinstance(metrics_dict[key], torch.Tensor)
                else metrics_dict[key]
            )
            history[key].append(metric_value)

        val_map = history["map"][-1]  # Use map@0.50:0.95 for early stopping
        val_iou = history["val_iou"][-1]
        LOGGER.info(
            f"Epoch {epoch+1}/{num_epochs}, Val mAP@0.50:0.95: {val_map:.4f}, Val IoU: {val_iou:.4f}"
        )

        # Check for improvement
        if early_stopping:
            if val_map > best_val_map:
                best_val_map = val_map
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                LOGGER.info(
                    f"New best model found at epoch {epoch+1} with Val mAP@0.50:0.95: {val_map:.4f}"
                )
            else:
                epochs_no_improve += 1
                LOGGER.info(f"No improvement for {epochs_no_improve} epoch(s).")
                if epochs_no_improve >= patience:
                    LOGGER.info("Early stopping triggered.")
                    break

    # Load best model state before returning
    model.load_state_dict(best_model_state)

    # Final evaluation on val set with best model
    model_metrics(val_data, wrapper)

    return history


def plot_history(
    history: dict, show: bool = True, save_path: Optional[str] = None
) -> None:
    """Plot the training loss and validation mAP metrics history.

    Args:
        show (bool, optional): Whether to show the plot immediately.
                                Defaults to True.
        save_path (Optional[str], optional): The path to save the plot to.
                                                Defaults to None.
    """
    # Create x-axis values for train and val metrics
    x_train = np.asarray(range(1, len(history["train_loss"]) + 1))

    # Get all metric keys (everything except train_loss)
    metric_keys = [k for k in history.keys() if k != "train_loss"]

    if not metric_keys:
        LOGGER.warning("No metrics found in history to plot.")
        return

    x_val = np.asarray(range(1, len(history[metric_keys[0]]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")

    # Plot 1: Training loss
    ax1.plot(
        x_train,
        history["train_loss"],
        label="Train Loss",
        color="tab:blue",
        linewidth=2,
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y")
    ax1.set_title("Training Loss")
    ax1.grid(True)

    # Plot 2: All validation mAP metrics
    for metric_key in metric_keys:
        ax2.plot(x_val, history[metric_key], label=metric_key, linewidth=2)

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("mAP")
    ax2.set_title("Validation mAP Metrics")
    fig.legend(loc="outside right upper", bbox_to_anchor=(1.05, 1), borderaxespad=0)
    ax2.grid(True)

    if save_path:
        # Keep outside-axes elements (e.g., figure-level legend) inside the output image.
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        LOGGER.info(f"Training history plot saved to {save_path}")

    if show:
        plt.show()


def model_metrics(test: Dataset, wrapper: FasterRCNNWrapper) -> None:
    # Get outputs from the model on the test set and visualize some predictions
    output_dir = Path("out") / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, targets = wrapper.get_predictions(test)

    # mAP
    map_metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", backend="faster_coco_eval"
    )
    map_metric = map_metric.to(DEVICE)

    map_metric.update(predictions, targets)
    metrics_dict = map_metric.compute()

    print_table(metrics_dict, title="Test Set Evaluation Metrics")

    # IoU distribution
    iou_values = []
    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"].cpu()
        target_boxes = target["boxes"].cpu()
        pairwise_iou = generalized_box_iou(pred_boxes, target_boxes)
        iou_values.extend(pairwise_iou.flatten().tolist())

    # table of IoU distribution
    iou_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    iou_hist, _ = np.histogram(iou_values, bins=iou_bins)
    iou_hist = iou_hist / np.sum(iou_hist)  # Normalize to get percentages
    iou_table_data = {
        f"{iou_bins[i]:.2f}-{iou_bins[i+1]:.2f}": f"{iou_hist[i]*100:.2f}%"
        for i in range(len(iou_bins) - 1)
    }
    print_table(iou_table_data, title="IoU Distribution on Test Set")
