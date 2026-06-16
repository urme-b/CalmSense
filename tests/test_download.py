"""Archive extraction refuses path traversal and oversized archives."""

import zipfile

import pytest

from scripts.download_data import _download, _safe_extract


def test_safe_extract_unpacks_benign_zip(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("a/b.txt", "hello")
    out = tmp_path / "out"
    _safe_extract(archive, out)
    assert (out / "a" / "b.txt").read_text() == "hello"


def test_safe_extract_blocks_zip_slip(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", "pwned")
    with pytest.raises(RuntimeError, match="unsafe path"):
        _safe_extract(archive, tmp_path / "out")


def test_safe_extract_blocks_zip_bomb(tmp_path):
    archive = tmp_path / "big.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("data.bin", b"x" * 4096)
    with pytest.raises(RuntimeError, match="size cap"):
        _safe_extract(archive, tmp_path / "out", max_bytes=1024)


def test_download_refuses_non_https(tmp_path):
    with pytest.raises(ValueError, match="non-HTTPS"):
        _download("http://example.com/x.zip", tmp_path / "x.zip")
