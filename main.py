from src.utils.download import download_data, cleanup_files
from src.const import DATA_PATH, NEON_TREE_PATH
from src.data.setup import SetupNeonTreeData

from pathlib import Path


def main():
    if not Path(DATA_PATH).exists():
        download_data(DATA_PATH)
    cleanup_files(NEON_TREE_PATH)
    data = SetupNeonTreeData()
    print("Data loaded successfully.")


if __name__ == "__main__":
    main()
