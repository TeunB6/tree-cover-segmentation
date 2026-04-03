from torch import Tensor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
)

# Weight enums for pretrained models
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights

from typing import Literal


class FasterRCNNWrapper:
    def __init__(
        self,
        num_classes: int,
        model_name: Literal[
            "resnet50", "mobilenet_v3_large_320", "mobilenet_v3_large"
        ] = "resnet50",
        pretrained: bool = False,
        pretrained_backbone: bool = True,
    ):
        # Initialize the Faster R-CNN model with the specified number of classes

        if model_name == "resnet50":
            self._model = fasterrcnn_resnet50_fpn(
                num_classes=num_classes,
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
                weights_backbone=(
                    ResNet50_Weights.DEFAULT if pretrained_backbone else None
                ),
            )
        elif model_name == "mobilenet_v3_large_320":
            self._model = fasterrcnn_mobilenet_v3_large_320_fpn(
                num_classes=num_classes,
                weights=(
                    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                    if pretrained
                    else None
                ),
                weights_backbone=(
                    MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
                ),
            )
        elif model_name == "mobilenet_v3_large":
            self._model = fasterrcnn_mobilenet_v3_large_fpn(
                num_classes=num_classes,
                weights=(
                    FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
                    if pretrained
                    else None
                ),
                weights_backbone=(
                    MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
                ),
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def __getattr__(self, name):
        # Delegate attribute access to the underlying model
        try:
            return self._model.__getattr__(name)
        except AttributeError:
            raise AttributeError(
                f"'FasterRCNNWrapper' object has no attribute '{name}'"
            )

    @classmethod
    def load(cls, model_path: str) -> "FasterRCNNWrapper":
        # Load a saved model from the specified path
        wrapper = cls(
            num_classes=2
        )  # num_classes is a placeholder, will be overwritten by loaded state
        wrapper._model.load_state_dict(Tensor.load(model_path))
        return wrapper
