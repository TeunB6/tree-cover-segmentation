from src.utils.singleton import SingletonMeta
from src.const import NEON_TREE_PATH, PT_DATA_PATH, LOGGER
import xmltodict
from torchvision.tv_tensors import BoundingBoxes
import torch
from skimage import io
from skimage.transform import resize as sk_resize
from src.utils.transforms import ToTensor


class SetupNeonTreeData(metaclass=SingletonMeta):
    def __init__(self, force_overwrite: bool = False):
        for split in ["train", "test"]:
            setattr(
                self, split, self._load_split(split, force_overwrite=force_overwrite)
            )

    def _load_split(self, split: str, force_overwrite: bool = False):
        if PT_DATA_PATH.exists() and not force_overwrite:
            # load data from pt_data
            return PT_DATA_PATH / split
        else:
            # load data from neon_tree
            PT_DATA_PATH.mkdir(parents=True, exist_ok=True)
            return self.create_split(split)

    def create_split(self, split: str):  # TODO: Add CHM and Hyperspectral data
        # Create torch data for the given split
        in_path = (
            NEON_TREE_PATH / "training"
            if split == "train"
            else NEON_TREE_PATH / "evaluation"
        ) / "RGB"

        # load images and labels, create torch data, save to pt_data
        LOGGER.log_and_print(f"Processing {split} split...")
        LOGGER.debug(
            f"Looking for images in: {in_path}\t{len(list(in_path.glob('*.tif')))} images found."
        )
        for img_path in in_path.glob("*.tif"):

            out_path = PT_DATA_PATH / split
            out_path.mkdir(parents=True, exist_ok=True)

            if out_path.joinpath(img_path.stem + ".pt").exists():
                LOGGER.info(f"Data for {img_path.name} already exists, skipping.")
                continue
            LOGGER.info(f"Processing image: {img_path.name}")
            label_path = (NEON_TREE_PATH / "annotations" / img_path.stem).with_suffix(
                ".xml"
            )
            chm_path = in_path.parent / "CHM" / (img_path.stem + "_CHM.tif")
            if not label_path.exists():
                LOGGER.warning(f"Missing annotation for {img_path.name}, skipping.")
                continue
            elif not chm_path.exists():
                LOGGER.warning(f"Missing CHM for {img_path.name}, skipping.")
                continue
            rgb = self._load_image(img_path)
            chm = self._load_image(chm_path, target_size=rgb.shape[1:])
            # Reshape to match dim
            LOGGER.debug(f"{rgb.shape}, {chm.shape}")
            bounding_boxes = self._load_bounding_boxes(
                label_path, canvas_size=rgb.shape[1:]
            )
            comb = torch.cat([rgb, chm], dim=0)
            LOGGER.debug(f"Combined tensor shape: {comb.shape}")

            # save comb and bounding_boxes to pt_data
            torch.save((comb, bounding_boxes), out_path / (img_path.stem + ".pt"))
        return out_path

    def _load_image(self, img_path, target_size: tuple[int, int] | None = None):
        img = io.imread(img_path)

        if target_size is not None:
            img = sk_resize(img, target_size, preserve_range=True).astype(img.dtype)

        return ToTensor()(img)

    def _load_bounding_boxes(self, label_path, canvas_size):
        # load bounding boxes from label_path
        with open(label_path) as f:
            data = xmltodict.parse(f.read())
        # extract bounding boxes from data
        if isinstance(data["annotation"]["object"], list):
            bounding_boxes = [
                [
                    int(object_dict["bndbox"]["xmin"]),
                    int(object_dict["bndbox"]["ymin"]),
                    int(object_dict["bndbox"]["xmax"]),
                    int(object_dict["bndbox"]["ymax"]),
                ]
                for object_dict in data["annotation"]["object"]
            ]
        elif isinstance(data["annotation"]["object"], dict):
            object_dict = data["annotation"]["object"]
            bounding_boxes = [
                [
                    int(object_dict["bndbox"]["xmin"]),
                    int(object_dict["bndbox"]["ymin"]),
                    int(object_dict["bndbox"]["xmax"]),
                    int(object_dict["bndbox"]["ymax"]),
                ]
            ]
        else:
            LOGGER.warning(
                f"No bounding boxes found in {label_path.name}.\t{data['annotation']}"
            )
        LOGGER.info(f"Extracted {len(bounding_boxes)} bounding boxes.")
        return BoundingBoxes(
            bounding_boxes, format="XYXY", dtype=torch.float32, canvas_size=canvas_size
        )
