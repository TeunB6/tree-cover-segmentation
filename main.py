from src.utils.download import download_data, cleanup_files
from src.const import DATA_PATH, NEON_TREE_PATH, LOGGER
from src.data.setup import SetupNeonTreeData
from src.data.dataset import TreeImageDataset
from src.utils.visual import view_image_with_boxes
from src.models.faster_rcnn import FasterRCNNWrapper
import matplotlib.pyplot as plt

from src.utils.cli import cli_menu

from pathlib import Path

from random import choice
from time import sleep

# Training
from src.models.trainer import train_faster_rcnn, plot_history
from torch.optim import AdamW


def data_inspection():
    dataset = TreeImageDataset(split="train")

    # Plot some random samples from the dataset
    for _ in range(5):
        idx = choice(range(len(dataset)))
        image, targets = dataset[idx]
        boxes = targets["boxes"]
        view_image_with_boxes(image, boxes, save_path=Path("out") / f"sample_{idx}.png")
        plt.close()


def train_model():
    fasterrcnn = FasterRCNNWrapper(num_classes=2, pretrained_backbone=True)
    train = TreeImageDataset(split="train")
    test = TreeImageDataset(split="test")
    val = TreeImageDataset(split="val")

    history = train_faster_rcnn(
        wrapper=fasterrcnn,
        train_data=train,
        val_data=val,
        test_data=test,
        num_epochs=10,
        optimizer=AdamW,
        early_stopping=True,
        patience=3,
    )

    plot_history(history, save_path=Path("out") / "training_history.png")

    # Save the best model state
    fasterrcnn.save_model(
        history["best_model_state"], save_path=Path("out") / "best_fasterrcnn.pth"
    )
    LOGGER.info(f"Best model saved to {Path('out') / 'best_fasterrcnn.pth'}")


def evaluate_model():
    # Get outputs from the model on the test set and visualize some predictions
    pass


def main():
    if not Path(DATA_PATH).exists():
        download_data(DATA_PATH)
    cleanup_files(NEON_TREE_PATH)
    data = SetupNeonTreeData(force_overwrite=False)
    print("Data loaded successfully.")

    cli_menu(
        "What would you like to do?",
        {
            "Data Inspection": data_inspection,
            "Train a Faster R-CNN model": train_model,
            "Evaluate the trained model": lambda: print(
                "Evaluation not implemented yet."
            ),
        },
    )


if __name__ == "__main__":
    main()
