from src.utils.download import download_data
from src.const import DATA_PATH

from pathlib import Path


def main():
    if not Path(DATA_PATH).exists():
        download_data(DATA_PATH)


if __name__ == "__main__":
    main()
