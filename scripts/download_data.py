"""Fetch the public datasets. Non-EEG downloads directly; WESAD needs a one-time
agreement, so we resolve its download link and unpack what you hand it."""

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR

NONEEG_URL = (
    "https://physionet.org/static/published-projects/noneeg/"
    "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0.zip"
)
NONEEG_DIR = EXTERNAL_DATA_DIR / "noneeg"
WESAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"


def _progress(block, block_size, total):
    done = block * block_size
    pct = min(100, 100 * done / total) if total > 0 else 0
    print(f"\r  {done // (1024 * 1024)} MB ({pct:.0f}%)", end="", flush=True)


def _download(url, dest):
    print(f"Downloading {url}")
    urlretrieve(url, dest, _progress)
    print()


def download_noneeg() -> None:
    target = NONEEG_DIR / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
    if target.exists():
        print(f"Non-EEG already present at {target}")
        return
    NONEEG_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = NONEEG_DIR / "noneeg.zip"
    _download(NONEEG_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(NONEEG_DIR)
    zip_path.unlink()
    print(f"Non-EEG ready at {target}")


def download_wesad() -> None:
    target = RAW_DATA_DIR / "WESAD"
    if (target / "S2" / "S2.pkl").exists():
        print(f"WESAD already present at {target}")
        return
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DATA_DIR / "WESAD.zip"
    print("WESAD is ~4 GB and covered by a research-only agreement.")
    _download(WESAD_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DATA_DIR)
    zip_path.unlink()
    print(f"WESAD ready at {target}")


def _check(url) -> None:
    with urlopen(url) as r:  # noqa: S310
        size = r.headers.get("Content-Length")
        print(f"{r.status}  {url}  ({int(size) // (1024 * 1024) if size else '?'} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wesad", action="store_true", help="also fetch WESAD (~4 GB)")
    parser.add_argument("--check", action="store_true", help="only verify the download links")
    args = parser.parse_args()

    if args.check:
        _check(NONEEG_URL)
        _check(WESAD_URL)
    else:
        download_noneeg()
        if args.wesad:
            download_wesad()
