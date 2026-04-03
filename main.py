from src.utils.download import download_data, cleanup_files
from src.const import DATA_PATH, NEON_TREE_PATH, PT_DATA_PATH
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
from src.models.trainer import Trainer
from torch.optim import AdamW
from src.utils.loss import GeneralizedBoxIoULoss


def data_inspection():
    dataset = TreeImageDataset(split="train")

    # Plot some random samples from the dataset
    for _ in range(4):
        idx = choice(range(len(dataset)))
        image, targets = dataset[idx]
        boxes = targets["boxes"]
        view_image_with_boxes(image, boxes)
        sleep(1)
        plt.close()


def train_model():
    model = FasterRCNNWrapper(num_classes=2, pretrained=False, pretrained_backbone=True)
    train = TreeImageDataset(split="train")
    test = TreeImageDataset(split="test")
    val = TreeImageDataset(split="val")

    trainer = Trainer(
        model=model._model,
        train_data=train,
        val_data=val,
        test_data=test,
        batch_size=64,
    )

    trainer.train(
        num_epochs=10,
        criterion=GeneralizedBoxIoULoss,
        optimizer=AdamW,
        early_stopping=True,
    )


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
        },
    )


if __name__ == "__main__":
    main()
