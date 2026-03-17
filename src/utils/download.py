from pathlib import Path
from src.const import DATA_URL, DATA_PATH
import requests
from zipfile import ZipFile


def download_data(save_path: Path | str, verbose: bool = True):
    save_path = Path(save_path)
    dir = save_path.parent if save_path.is_file() else save_path
    dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Downloading data from {DATA_URL} to {dir}...")
    # Get Zip file
    response = requests.get(DATA_URL, stream=True)
    temp_file = save_path / "temp.zip"
    total = int(response.headers.get("content-length", 0))
    chunk_size = 8192
    with open(temp_file, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if verbose and total:
                    done = int(50 * downloaded / total)
                    print(
                        "\r[{}{}] {:.1f}%".format(
                            "=" * done, " " * (50 - done), 100 * downloaded / total
                        ),
                        end="",
                    )
        if verbose and total:
            print()
    if verbose:
        print("Download Status:", response.status_code)
    temp_file = save_path / "temp.zip"

    with open(temp_file, "wb") as f:
        f.write(response.content)

    # Unzip file
    with ZipFile(temp_file, "r") as zip_ref:
        zip_ref.extractall(dir)

    # Remove temp file
    temp_file.unlink()


if __name__ == "__main__":
    download_data(DATA_PATH)
