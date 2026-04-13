from src.utils.download import download_data, cleanup_files
from src.const import DATA_PATH, NEON_TREE_PATH, LOGGER
from src.data.setup import SetupNeonTreeData
from src.data.dataset import TreeImageDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.utils.visual import view_image_with_boxes, view_prediction
from src.models.faster_rcnn import FasterRCNNWrapper
import matplotlib.pyplot as plt

from src.utils.cli import cli_menu
from rich.progress import track

from pathlib import Path

from random import choice

from datetime import datetime
import json

# Training
from src.models.trainer import train_faster_rcnn, plot_history, model_metrics
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


def train_model(transforms: bool = True):
    fasterrcnn = FasterRCNNWrapper(num_classes=2, pretrained_backbone=True)
    train = TreeImageDataset(
        split="train",
        transforms=get_train_transforms() if transforms else None,
        force_lazy_loading=True,
        # transform_inflate_factor=3 if transforms else 1,  # Inflate dataset by applying random transforms
    )
    test = TreeImageDataset(
        split="test", transforms=get_val_transforms() if transforms else None
    )
    val = TreeImageDataset(
        split="val", transforms=get_val_transforms() if transforms else None
    )

    history = train_faster_rcnn(
        wrapper=fasterrcnn,
        train_data=train,
        val_data=val,
        num_epochs=50,
        optimizer=AdamW,
        early_stopping=True,
        patience=4,
    )

    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Plot training history
    output_dir = Path("out") / "training_history"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_history(history, save_path=output_dir / f"training_history_{time_stamp}.png")

    # Save history as json for later analysis
    with open(output_dir / f"training_history_{time_stamp}.json", "w") as f:
        json.dump(history, f, indent=4)

    # Save the best model state
    output_dir = Path("out") / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    fasterrcnn.save_model(
        save_path=output_dir
        / f"best_fasterrcnn_{"transformed" if transforms else "untransformed"}_{time_stamp}.pth",
    )


def evaluate_model():
    # Get outputs from the model on the test set and visualize some predictions
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("out") / "predictions" / time_stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find most recent model checkpoint
    model_dir = Path("out") / "models"
    model_files = sorted(
        model_dir.glob("best_fasterrcnn_*.pth"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not model_files:
        LOGGER.error(
            "No model checkpoints found in 'out/models'. Please train a model first."
        )
        return
    latest_model_path = model_files[0]
    LOGGER.info(f"Loading model from {latest_model_path}")
    fasterrcnn = FasterRCNNWrapper.load(latest_model_path)
    fasterrcnn.eval()

    test = TreeImageDataset(split="test", transforms=get_val_transforms())
    metrics = model_metrics(test, fasterrcnn)

    # Save metrics to json
    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    predictions, targets = fasterrcnn.get_predictions(test)

    LOGGER.info(f"Got predictions for {len(predictions)}/{len(targets)} test samples.")
    LOGGER.debug(f"Targets example: {targets[0]}")

    for idx, (image, targets) in track(
        enumerate(test), description="Visualizing predictions:", total=len(test)
    ):
        pred_boxes = predictions[idx]["boxes"]
        target_boxes = targets["boxes"]
        view_prediction(
            image,
            pred_boxes,
            target_boxes,
            save_path=output_dir / f"prediction_{test.get_site_name(idx)}_{idx}.png",
            show=False,
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
            "Train a Faster R-CNN model (transformed)": train_model,
            "Train a Faster R-CNN model (untransformed)": lambda: train_model(
                transforms=False
            ),
            "Evaluate the trained model": evaluate_model,
        },
    )


if __name__ == "__main__":
    LOGGER.info("Starting Single Tree Detection Project")
    main()
