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

# SHA-256 of the official WESAD S*.pkl files (also documented in data/raw/README.md).
WESAD_SHA256 = {
    "S2": "36ef5e8afc0f91998eefba7c12fc9fa97b7b07198cbec0126917d7abb436ca23",
    "S3": "5c8bd4a82af029c082e610bca28a011fca2ae3b23e14a18458ebb5990be4015e",
    "S4": "0f0740a79388723360ff12b4f47c465665ea7827d1399b18ac43908daac17900",
    "S5": "74bd187e3a9c1ca4259af52d04974c8e7ff7dc49ceea7e269f499ca98fe6d8ec",
    "S6": "8aa9bf57b69f4fe5bce06c550230857627c3f05befa2f787151646bb29ee8f62",
    "S7": "9cb62705ae7f53dca327a9a00a6f9fdabf5128d449174ab37594658e912cb6d8",
    "S8": "dac1141dac11d56b3641be982f45da63f05e9d74154f59e6ea0cdcf47fc72710",
    "S9": "24dc004e201bd541f092989443f0a29ebf89e4a227a80bb6b6d1987255039544",
    "S10": "41da29c68366f33650f3d41a6be78107bf6942929c3bb0ef46238078ddddee9f",
    "S11": "f39557a8d660b10154936f51debf2926aea7ebb9b26a168858f59502f914d8f7",
    "S13": "772fb490f19b279e49367271e009fc10d3a3ca1e3456df0d68b9063a73992066",
    "S14": "e7bd33c57538319a25c6d53e6a9fb6c1abd12800cfc64bb63275d89de8d2fd60",
    "S15": "1ea573bc6b45ba79fb134f9460d691b86176f60dce23420dc514c28017d4049c",
    "S16": "f65cf40cada75c3e9f5813276d7dcc90359c3b06dec41d68656c0a6e61dbc575",
    "S17": "3315796a75227d54d7b0056736f671484fd2fb85afffa65818fd76aeff2920fa",
}


def verify_wesad() -> None:
    """Check each downloaded WESAD S*.pkl against its known SHA-256; fail loudly on mismatch."""
    import hashlib

    root = RAW_DATA_DIR / "WESAD"
    problems = []
    for sid, expected in WESAD_SHA256.items():
        path = root / sid / f"{sid}.pkl"
        if not path.exists():
            problems.append(f"{sid}: missing")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            problems.append(f"{sid}: checksum mismatch")
    if problems:
        raise SystemExit(
            "WESAD integrity check FAILED:\n  "
            + "\n  ".join(problems)
            + "\nRe-download from the official source (see data/raw/README.md)."
        )
    print(f"WESAD integrity OK: {len(WESAD_SHA256)} subjects verified.")


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
    verify_wesad()
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
    parser.add_argument(
        "--verify-wesad", action="store_true", help="check downloaded WESAD .pkl SHA-256 and exit"
    )
    args = parser.parse_args()

    if args.verify_wesad:
        verify_wesad()
    elif args.check:
        _check(NONEEG_URL)
        _check(WESAD_URL)
    elif not args.wesad and not args.noneeg:
        download_noneeg()  # default with no flags, keeps `make data` fetching Non-EEG
    else:
        if args.noneeg:
            download_noneeg()
        if args.wesad:
            download_wesad()
