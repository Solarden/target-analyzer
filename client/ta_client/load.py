"""Decoding a photo into pixels, metadata-free bytes, and its identity hash.
See internal_docs/implementation.md §9 and §12.4.

The sha256 is taken of the *re-encoded* bytes, never the file on disk: that is what
makes it safe to use as the wire identity, because no GPS or camera EXIF survives into
the thing being hashed or shipped.
"""

import hashlib
import subprocess  # nosec B404
import tempfile
from pathlib import Path

import cv2
import numpy as np

SIPS = "/usr/bin/sips"
# Pinned rather than left to OpenCV's default so the identity hash of a given photo does
# not move if that default ever changes.
JPEG_QUALITY = 95


def _header(path: Path) -> bytes:
    """The first bytes of the file, or a readable error saying why there are none."""
    try:
        with path.open("rb") as handle:
            return handle.read(12)
    except OSError as exc:
        # Inside the Photos library this is a permissions refusal, which OpenCV would
        # report as an undecodable image. Export the photo out of the library instead.
        raise ValueError(f"cannot read {path}: {exc}") from exc


def _is_iso_bmff(header: bytes) -> bool:
    """Whether the file is an ISO-BMFF container — HEIC/HEIF, as the iPhone shoots."""
    return header[4:8] == b"ftyp"


def _decode_via_sips(path: Path) -> np.ndarray:
    """Decode what OpenCV could not, using macOS's own image tool.

    OpenCV has no HEIC decoder and the iPhone's default format is HEIC, so without this
    every photo needs a manual round-trip through Preview first.
    """
    if not _is_iso_bmff(_header(path)):
        raise ValueError(f"cannot decode {path} — expected JPEG, PNG, WebP or HEIC")

    with tempfile.TemporaryDirectory() as tmp:
        converted = Path(tmp) / "converted.jpg"
        # ponytail: macOS-only, which the Mac client already is. A cross-platform client
        # would need pillow-heif and the dependency that comes with it.
        # Fixed absolute binary, argv list, no shell. Everything after nosec is parsed as
        # a test id, so the reason for it lives here rather than on that line.
        done = subprocess.run(  # nosec B603
            [SIPS, "-s", "format", "jpeg", str(path), "--out", str(converted)],
            capture_output=True,
            check=False,
        )

        if done.returncode != 0:
            raise ValueError(f"sips could not convert {path}: {done.stderr.decode().strip()}")

        bgr = cv2.imread(str(converted), cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError(f"cannot decode {path} even after converting it to JPEG")

    return bgr


def load_stripped(path: str | Path) -> tuple[np.ndarray, bytes, str]:
    """Return ``(pixels, metadata-free JPEG bytes, sha256 of those bytes)``."""
    path = Path(path)
    _header(path)  # fail on a missing or unreadable file before OpenCV blames the format

    # No IMREAD_IGNORE_ORIENTATION: imread applies the EXIF rotation while decoding, and
    # that is what makes a portrait phone photo load upright rather than sideways.
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if bgr is None:
        bgr = _decode_via_sips(path)

    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    if not ok:
        raise ValueError(f"could not re-encode {path}")

    jpeg = buffer.tobytes()

    return bgr, jpeg, hashlib.sha256(jpeg).hexdigest()
