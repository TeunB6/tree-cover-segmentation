from pathlib import Path
from zipfile import ZipFile
import hashlib

import requests
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)

from src.const import DATA_URL, DATA_PATH, DATA_HASH
import shutil


def download_data(save_path: Path | str, verbose: bool = True):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    temp_file = save_path / "temp.zip"

    if not temp_file.exists():
        # Download and hash in one pass
        response = requests.get(DATA_URL, stream=True)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        h = hashlib.sha256()

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            disable=not verbose,
        ) as progress:
            task = progress.add_task("Downloading", total=total or None)

            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    h.update(chunk)
                    progress.advance(task, len(chunk))

        # Verify checksum
        digest = h.hexdigest()
        if digest != DATA_HASH:
            temp_file.unlink()
            raise ValueError(f"Hash mismatch: expected {DATA_HASH}, got {digest}")
        if verbose:
            print("Checksum verified.")

    # Recursively unzip
    _unzip_recursive(temp_file, save_path, verbose=verbose)
    temp_file.unlink()
    
    cleanup_files(save_path / "neon_tree" / "NeonTreeEvaluation")


def _unzip_recursive(zip_path: Path, dest: Path, verbose: bool = True):
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    for extracted in dest.rglob("*.zip"):
        if extracted == zip_path:
            continue
        if verbose:
            print(f"Unzipping nested: {extracted.relative_to(dest)}")
        _unzip_recursive(extracted, extracted.parent / extracted.stem, verbose=verbose)
        extracted.unlink()

def cleanup_files(save_path: Path | str):
    save_path = Path(save_path)
    
    def remove_nested(save_path: Path, target_name: str):
        dir = save_path / target_name
        nested_dir = dir / target_name
        if nested_dir.exists():
            for item in nested_dir.iterdir():
                shutil.move(str(item), dir / item.name)
            nested_dir.rmdir()
    
    # Move evaluation/evaluation to evaluation/ and remove
    remove_nested(save_path, "evaluation")
    remove_nested(save_path, "annotations")
    