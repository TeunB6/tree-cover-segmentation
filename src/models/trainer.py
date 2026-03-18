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
from typing import Optional
from pathlib import Path
from rich.progress import Progress

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


class Trainer:
    """Class for a trainer for NLP."""

    def __init__(
        self,
        model: nn.Module,
        train_data: Dataset,
        dev_data: Dataset,
        test_data: Optional[Dataset] = None,
        batch_size: int = 32,
    ) -> None:
        """Initialize the trainer.

        Args:
            model (nn.Module): The model.
            train_data (Dataset): The training data.
            eval_data (Dataset): The evaluation data.
            batch_size (int, optional): The batch size. Defaults to 32.
        """
        self.model = model.to(DEVICE)
        self.batch_size = batch_size
        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
        )

        self.dev_loader = DataLoader(
            dev_data,
            batch_size=batch_size,
            shuffle=False,
        )

        if test_data is not None:
            self.test_loader = DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            self.test_loader = None

    def reset_history(self):
        """Reset the training history."""
        self.history = {"train_loss": [], "eval_loss": []}

    def train(
        self,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        optimizer: optim.Optimizer = optim.Adam,
        criterion: nn.Module = nn.CrossEntropyLoss,
        early_stopping: bool = False,
        patience: int = 3,
    ) -> None:
        """Train the model.

        Args:
            num_epochs (int, optional): The number of epochs to train for.
                                        Defaults to 10.
            learning_rate (float, optional): The learning rate to use for
                                             optimization. Defaults to 1e-3.
            config (dict, optional): The configuration for the model. Defaults
                                     to {}.
            early_stopping (bool, optional): Whether to use early stopping for
                                             the training. Defaults to False.
            patience (int, optional): The number of epochs to wait for
                                      improvement, if early stopping is
                                      enabled. Defaults to 3.
        """
        self.reset_history()

        # Set up optimizer and loss function.
        optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        criterion = criterion().to(DEVICE)

        if early_stopping:
            best_eval_loss = float("inf")
            best_model_state = None
            epochs_no_improve = 0

        with Progress() as progress:
            epoch_task = progress.add_task("[cyan]Training Epochs", total=num_epochs)
            
            for epoch in range(num_epochs):
                self.model.train()

                total_loss = 0
                batch_task = progress.add_task("[magenta]Training Batches", total=len(self.train_loader))
                
                for batch in self.train_loader:
                    # Update model parameters based on the batch.
                    inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

                    optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    self.history["train_loss"].append(loss.item())
                    total_loss += loss.item()
                    
                    progress.update(batch_task, advance=1)

                avg_loss = total_loss / len(self.train_loader)
                eval_loss = self.evaluate(criterion)

                self.history["eval_loss"].append(eval_loss)

                progress.update(
                    epoch_task,
                    advance=1,
                    description=f"[cyan]Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Eval Loss: {eval_loss:.4f}"
                )

                if early_stopping:
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        epochs_no_improve = 0
                        best_model_state = copy.deepcopy(self.model.state_dict())
                    else:
                        epochs_no_improve += 1
                        LOGGER.info(f"No improvement in validation loss for {epochs_no_improve} epochs.")

                    if epochs_no_improve >= patience:
                        panel = Panel(
                        f"[bold red]Early stopping triggered after {epochs_no_improve+1} epochs without improvement![/bold red]"
                        )
                        LOGGER.log_and_print(panel)
                        break

        if early_stopping and best_model_state is not None:
            # If early stopping occurs, set the best model weights before the
            # end of training.
            self.model.load_state_dict(best_model_state)
            LOGGER.log_and_print(
                "Restored the best model weights from early stopping."
            )

    def evaluate(
        self,
        criterion: nn.Module,
        use_predict: bool = False,
        use_test: bool = False,
    ) -> float:
        """Evaluate the model on the evaluation set.

        Args:
            criterion (nn.Module): The loss function.

        Returns:
            float: The average loss on the evaluation set.
        """
        self.model.eval()

        total_loss = 0

        if use_test and self.test_loader is None:
            LOGGER.warning(
                "Test loader is not available. Evaluating on dev set instead."
            )
            loader = self.dev_loader
        elif use_test and self.test_loader is not None:
            loader = self.test_loader
        else:
            loader = self.dev_loader

        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                outputs = (
                    self.model.predict(inputs, return_prob=False)
                    if use_predict
                    else self.model(inputs)
                )
                loss = criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(loader)

    def plot_history(
        self, show: bool = True, save_path: Optional[str] = None
    ) -> None:
        """Plot the training and evaluation loss history.

        Args:
            show (bool, optional): Whether to show the plot immediately.
                                   Defaults to True.
            save_path (Optional[str], optional): The path to save the plot to.
                                                 Defaults to None.
        """
        # Create x-axis values for train and eval loss based on the number of recorded losses.
        x_train = np.asarray(range(1, len(self.history["train_loss"]) + 1))
        x_eval = np.asarray(range(1, len(self.history["eval_loss"]) + 1))
        x_train = x_train * (
            len(self.history["eval_loss"]) / len(self.history["train_loss"])
        )

        plt.figure()
        plt.plot(x_train, self.history["train_loss"], label="Train Loss")
        plt.plot(x_eval, self.history["eval_loss"], label="Eval Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss History")
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            LOGGER.info(f"Training history plot saved to {save_path}")

        if show:
            plt.show()

    def save_model(self, path: str | Path) -> None:
        """Save the model.

        Args:
            path (str): The path to save the model to.
        """
        torch.save(self.model.state_dict(), path)
        LOGGER.info(f"Model saved to {path}")

    def load_model(self, path: str | Path) -> None:
        """Load the model.

        Args:
            path (str): The path to load the model from.
        """

        self.model.load_state_dict(torch.load(path))
        self.model.to(DEVICE)
        LOGGER.info(f"Model loaded from {path}")