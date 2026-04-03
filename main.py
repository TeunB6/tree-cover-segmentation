from src.utils.download import download_data, cleanup_files
from src.const import DATA_PATH, NEON_TREE_PATH, PT_DATA_PATH
from src.data.setup import SetupNeonTreeData
from src.utils.visual import view_image_with_boxes_from_name

from pathlib import Path

from random import choice


def main():
    if not Path(DATA_PATH).exists():
        download_data(DATA_PATH)
    cleanup_files(NEON_TREE_PATH)
    data = SetupNeonTreeData(force_overwrite=False)
    print("Data loaded successfully.")
    
    
    # Get random sample images and visualize them with bounding boxes
    list_of_images = list((PT_DATA_PATH / "test" / "images").glob("*.tif"))

    for _ in range(10):
        random_image = choice(list_of_images).stem
        print(f"Viewing image: {random_image}")
        
        view_image_with_boxes_from_name(random_image, split="test")


if __name__ == "__main__":
    main()
