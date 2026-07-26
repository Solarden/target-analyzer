"""The upload trust boundary (implementation.md §7.4-§7.6, §12 items 3-4)."""

from io import BytesIO

import pytest
from PIL import Image as PILImage
from tests.conftest import make_jpeg, make_png

from target_analyzer import storage


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (make_jpeg(), "image/jpeg"),
        (make_png(8), "image/png"),
        (b"RIFF\x24\x00\x00\x00WEBPVP8 ", "image/webp"),
        # Everything below is what an attacker actually sends: a real file type wearing
        # an image's filename. The declared Content-Type is never consulted, so the
        # only thing standing here is the signature.
        (b"MZ\x90\x00", None),  # PE executable
        (b"\x7fELF\x02\x01\x01", None),  # ELF binary
        (b"<html><body>hi</body></html>", None),
        (b"PK\x03\x04", None),  # zip / docx container
        (b"%PDF-1.7", None),  # allowed in EA, not here
        (b"RIFF\x24\x00\x00\x00WAVEfmt ", None),  # RIFF, but not WebP
        (b"", None),
    ],
)
def test_sniff_content_type(data, expected):
    assert storage.sniff_content_type(data) == expected


def test_decode_verified_reports_size():
    assert storage.decode_verified(make_png(32)) == (32, 32)


@pytest.mark.parametrize(
    "data",
    [
        b"\xff\xd8\xff" + b"not really a jpeg",  # right magic bytes, garbage body
        make_png(16)[:40],  # truncated mid-file
    ],
)
def test_decode_verified_rejects_undecodable(data):
    """A sniffed signature proves four bytes; only a decode proves an image."""
    with pytest.raises(ValueError):
        storage.decode_verified(data)


def test_decode_verified_rejects_a_decompression_bomb(monkeypatch):
    """A small file may declare a huge canvas — the cap must bite before rasterising."""
    monkeypatch.setattr(storage, "MAX_PIXELS", 100)

    with pytest.raises(ValueError, match="pixel cap"):
        storage.decode_verified(make_png(64))


def test_strip_metadata_removes_exif():
    """GPS from a phone photo of a shooting spot must not reach the disk (§12 item 4)."""
    original = make_jpeg(with_exif=True)
    assert PILImage.open(BytesIO(original)).getexif()

    stripped = storage.strip_metadata(original, "image/jpeg")

    assert not PILImage.open(BytesIO(stripped)).getexif()


def test_store_generates_its_own_name(tmp_path):
    """On-disk names are UUIDs, never client input — the reason traversal can't start."""
    rel = storage.store(tmp_path, "originals", make_jpeg(), "image/jpeg")

    assert rel.startswith("originals/")
    assert rel.endswith(".jpg")
    assert (tmp_path / rel).exists()


def test_delete_tolerates_a_missing_file(tmp_path):
    """Rollback unlinks blind — it must not fail on a file that was never written."""
    storage.delete(tmp_path, "originals/never-existed.jpg")
