from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch
from rich.progress import track
from rich.panel import Panel
import copy
from src.const import LOGGER, DEVICE
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable
from pathlib import Path
from rich.progress import Progress

from src.models.faster_rcnn import FasterRCNNWrapper
from src.utils.misc import GeneralizedBoxIoULoss, detection_collate_fn

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
    test_data: Dataset,
    num_epochs: int = 10,
    criterion: Callable = GeneralizedBoxIoULoss,
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
        criterion (Callable, optional): The loss function to use. Defaults to GeneralizedBoxIoULoss.
        optimizer (Callable, optional): The optimizer to use. Defaults to optim.AdamW.
        early_stopping (bool, optional): Whether to use early stopping. Defaults to True.
        patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 3.

    Returns:
        dict: A dictionary containing the training history and the best model state.
    """
    # Move model to device
    model = wrapper.model
    model.to(DEVICE)

    # Create data loaders
    train_loader = DataLoader(
        train_data, batch_size=16, shuffle=True, collate_fn=detection_collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size=16, shuffle=False, collate_fn=detection_collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=16, shuffle=False, collate_fn=detection_collate_fn
    )

    # Initialize optimizer and loss function
    optim = optimizer(model.parameters())
    criterion = criterion()

    history = {"train_loss": [], "eval_loss": []}
    best_model_state = copy.deepcopy(model.state_dict())
    best_eval_loss = float("inf")
    epochs_no_improve = 0

    for epoch in track(range(num_epochs), description="Training..."):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for images, targets in train_loader:
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optim.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optim.step()

            total_train_loss += losses.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        LOGGER.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
    
        # Evaluation phase
        model.eval()
        total_eval_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_eval_loss += losses.item()
        avg_eval_loss = total_eval_loss / len(val_loader)
        history["eval_loss"].append(avg_eval_loss)
        LOGGER.info(f"Epoch {epoch+1}/{num_epochs}, Eval Loss: {avg_eval_loss:.4f}")
    
        # Check for improvement
        if early_stopping:
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                LOGGER.info(f"New best model found at epoch {epoch+1} with Eval Loss: {avg_eval_loss:.4f}")
            else:
                epochs_no_improve += 1
                LOGGER.info(f"No improvement for {epochs_no_improve} epoch(s).")
                if epochs_no_improve >= patience:
                    LOGGER.info("Early stopping triggered.")
                    break
    
    # Load best model state before returning
    model.load_state_dict(best_model_state)
    return history
    



def plot_history(history: dict, show: bool = True, save_path: Optional[str] = None) -> None:
    """Plot the training and evaluation loss history.

    Args:
        show (bool, optional): Whether to show the plot immediately.
                                Defaults to True.
        save_path (Optional[str], optional): The path to save the plot to.
                                                Defaults to None.
    """
    # Create x-axis values for train and eval loss based on the number of recorded losses.
    x_train = np.asarray(range(1, len(history["train_loss"]) + 1))
    x_eval = np.asarray(range(1, len(history["eval_loss"]) + 1))
    x_train = x_train * (
        len(history["eval_loss"]) / len(history["train_loss"])
    )

    plt.figure()
    plt.plot(x_train, history["train_loss"], label="Train Loss")
    plt.plot(x_eval, history["eval_loss"], label="Eval Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss History")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        LOGGER.info(f"Training history plot saved to {save_path}")

    if show:
        plt.show()

def save_model(model, path: str | Path) -> None:
    """Save the model.

    Args:
        path (str): The path to save the model to.
    """
    torch.save(model.state_dict(), path)
    LOGGER.info(f"Model saved to {path}")
