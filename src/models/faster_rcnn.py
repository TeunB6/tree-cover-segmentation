import torch
from collections import OrderedDict
from torchvision.models.detection import (
    FasterRCNN,
)
from torch.utils.data import DataLoader, Dataset
from src.utils.misc import detection_collate_fn
from rich.progress import track

# Weight enums for pretrained models
from torchvision.models import resnet50, ResNet50_Weights

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import BackboneWithFPN

from torchvision.models.detection.rpn import AnchorGenerator

from typing import Literal
from src.const import CHANNEL_MEANS, CHANNEL_STDEVS, LOGGER, DEVICE


class FasterRCNNWrapper:
    def __init__(
        self,
        num_classes: int,
        model_name: Literal[
            "resnet50"  # , "mobilenet_v3_large_320", "mobilenet_v3_large" # TODO: add more backbone options
        ] = "resnet50",
        pretrained_backbone: bool = True,
    ):
        # Initialize the Faster R-CNN model with the specified number of classes
        # Anchor sizes scaled for 400x400 images at different FPN levels
        # Sizes should increase with feature map level (lower resolution = larger receptive field)
        anchor_generator = AnchorGenerator(
            sizes=(
                (32, 64, 128),  # FPN level 0 (stride 4)
                (64, 128, 256),  # FPN level 1 (stride 8)
                (128, 256, 512),  # FPN level 2 (stride 16)
                (256, 512, 1024),  # FPN level 3 (stride 32)
                (512, 1024, 1024),  # FPN level pool (stride 64)
            ),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3", "pool"], output_size=7, sampling_ratio=2
        )

        self.model = FasterRCNN(
            backbone=self._get_backbone(
                model_name, pretrained_backbone
            ),  # We will set the backbone separately
            num_classes=num_classes,
            image_mean=CHANNEL_MEANS,
            image_std=CHANNEL_STDEVS,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

    def _get_backbone(
        self, model_name: str, pretrained_backbone: bool, n_channels: int = 4
    ) -> torch.nn.Module:
        if model_name == "resnet50":
            backbone = resnet50(
                weights=ResNet50_Weights.DEFAULT if pretrained_backbone else None
            )  # 5 is prob too much

            # Adapt the backbone to n-channel input if necessary (e.g., for 4-channel input)
            if backbone.conv1.in_channels != n_channels:
                old_weight = backbone.conv1.weight.data.clone()
                backbone.conv1 = torch.nn.Conv2d(
                    n_channels,
                    backbone.conv1.out_channels,
                    kernel_size=backbone.conv1.kernel_size,
                    stride=backbone.conv1.stride,
                    padding=backbone.conv1.padding,
                    bias=backbone.conv1.bias is not None,
                )

                # If using pretrained weights, we can copy the weights for the first 3 channels and initialize the new channel randomly
                if pretrained_backbone:
                    with torch.no_grad():
                        backbone.conv1.weight[:, :3, :, :] = old_weight
                        if n_channels > 3:
                            # Initialize new channels with small random values, scaled by 1/n_channels for stability
                            backbone.conv1.weight[:, 3:, :, :].normal_(
                                0, 0.01 / n_channels
                            )

        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        LOGGER.info(
            f"Using {model_name} backbone with {'pretrained' if pretrained_backbone else 'random'} weights: {type(backbone)}"
        )

        backbone = BackboneWithFPN(
            backbone,
            return_layers={"layer1": 0, "layer2": 1, "layer3": 2, "layer4": 3},
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
        )

        # Log backbone output shapes
        self._log_backbone_output_shape(backbone, n_channels)

        return backbone

    def _log_backbone_output_shape(self, backbone: torch.nn.Module, n_channels: int):
        """Create a dummy input and log the output shapes from the backbone."""
        dummy_input = torch.randn(1, n_channels, 400, 400)
        with torch.no_grad():
            outputs = backbone(dummy_input)

        LOGGER.info(f"Backbone output shapes for input {dummy_input.shape}:")
        for key, value in outputs.items():
            LOGGER.info(f"  Feature map '{key}': {value.shape}")

    def __getattr__(self, name):
        # Delegate attribute access to the underlying model
        try:
            return getattr(self.model, name)
        except AttributeError:
            raise AttributeError(
                f"'FasterRCNNWrapper' object has no attribute '{name}'"
            )
    
    def get_predictions(self, data: Dataset) -> tuple[list, list]:
        loader = DataLoader(data, batch_size=16, shuffle=False, collate_fn=detection_collate_fn)
        self.model.to(DEVICE)
        self.model.eval()
        predictions = []
        full_targets = []
        with torch.no_grad():
            for images, targets in track(loader, description="Getting predictions:", total=len(loader)):
                batch_preds = self.model(images)
                predictions.extend(batch_preds)
                full_targets.extend(targets)
        return predictions, full_targets
    
    @classmethod
    def load(cls, model_path: str) -> "FasterRCNNWrapper":
        # Load a saved model from the specified path
        wrapper = cls(
            num_classes=2
        )  # num_classes is a placeholder, will be overwritten by loaded state
        wrapper.model.load_state_dict(
            torch.load(model_path)
        )  # TODO: test if this works with the wrapper structure
        return wrapper

    def save_model(self, save_path: str) -> None:
        # Save the model state to the specified path
        torch.save(self.model.state_dict(), save_path)
        LOGGER.info(f"Model saved to {save_path}")
