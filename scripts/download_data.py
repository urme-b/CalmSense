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
MAX_UNCOMPRESSED = 10 * 1024**3  # zip-bomb guard


def _progress(block, block_size, total):
    done = block * block_size
    pct = min(100, 100 * done / total) if total > 0 else 0
    print(f"\r  {done // (1024 * 1024)} MB ({pct:.0f}%)", end="", flush=True)


def _download(url, dest):
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS download: {url}")
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    try:
        urlretrieve(url, part, _progress)
        part.replace(dest)  # only a complete download lands on the final path
    finally:
        part.unlink(missing_ok=True)
    print()


def _safe_extract(zip_path, dest, max_bytes=MAX_UNCOMPRESSED):
    """Extract, rejecting members that escape dest (zip-slip) or balloon (zip-bomb)."""
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as z:
        total = 0
        for info in z.infolist():
            resolved = (dest / info.filename).resolve()
            if resolved != dest and dest not in resolved.parents:
                raise RuntimeError(f"unsafe path in archive: {info.filename}")
            total += info.file_size
            if total > max_bytes:
                raise RuntimeError("archive expands beyond the size cap; refusing to extract")
        z.extractall(dest)


def download_noneeg() -> None:
    target = NONEEG_DIR / "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"
    if target.exists():
        print(f"Non-EEG already present at {target}")
        return
    NONEEG_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = NONEEG_DIR / "noneeg.zip"
    _download(NONEEG_URL, zip_path)
    _safe_extract(zip_path, NONEEG_DIR)
    zip_path.unlink()
    print(f"Non-EEG ready at {target}")


def download_wesad() -> None:
    target = RAW_DATA_DIR / "WESAD"
    if (target / "S2" / "S2.pkl").exists():
        print(f"WESAD already present at {target}")
        return
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DATA_DIR / "WESAD.zip"
    print("WESAD is ~2 GB and covered by a research-only agreement.")
    _download(WESAD_URL, zip_path)
    _safe_extract(zip_path, RAW_DATA_DIR)
    zip_path.unlink()
    print(f"WESAD ready at {target}")


def _check(url) -> None:
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS request: {url}")
    with urlopen(url) as r:
        size = r.headers.get("Content-Length")
        print(f"{r.status}  {url}  ({int(size) // (1024 * 1024) if size else '?'} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch CalmSense datasets.")
    parser.add_argument("--wesad", action="store_true", help="fetch WESAD (~2 GB, primary dataset)")
    parser.add_argument(
        "--noneeg", action="store_true", help="fetch PhysioNet Non-EEG (cross-dataset transfer)"
    )
    parser.add_argument("--check", action="store_true", help="only verify the download links")
    args = parser.parse_args()

    if args.check:
        _check(NONEEG_URL)
        _check(WESAD_URL)
    elif not args.wesad and not args.noneeg:
        download_noneeg()  # default with no flags, keeps `make data` fetching Non-EEG
    else:
        if args.noneeg:
            download_noneeg()
        if args.wesad:
            download_wesad()
