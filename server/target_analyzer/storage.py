"""On-disk image storage and the upload trust boundary (implementation.md §7.4-§7.6, §12).

Derived from expense-analyzer's ``attachments.py``; the two rules it bakes in carry over
unchanged, and a third is added because these files are images rather than documents:

- **Trust the bytes, not the client.** A file's type is decided by sniffing its magic
  bytes (:func:`sniff_content_type`), never by the declared ``Content-Type``. Anything
  that does not match an allowed signature is rejected.
- **Generated names, never user input.** The on-disk name is a fresh UUID plus the
  type's canonical extension, so a crafted filename cannot traverse out of the store.
  The client never sends a filename anyway — this is the reason it cannot start.
- **Decode before trusting, and bound the decode.** A sniffed header proves four bytes,
  not a valid image, so every upload is decode-verified; and the decode itself is
  capped (:data:`MAX_PIXELS`) because a 100 KB PNG can legally declare a 40000x40000
  canvas and eat all the Pi's RAM on the way to being rejected.

Paths handed back and persisted are **relative** to the data directory, so moving
``TA_DATA_PATH`` does not invalidate every row.
"""

import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Sniff-signature -> canonical extension. This map is the security allowlist (kept in
# code, not env): an upload must match one of these *by content* to be accepted. No PDF
# — unlike EA's loan documents, everything here is a photograph or a rendered target.
ALLOWED_IMAGE_TYPES: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

# Decompression-bomb ceiling. A 48 MP phone photo is ~48e6; 64e6 leaves headroom for a
# high-res scan while staying far under what would exhaust a Raspberry Pi.
MAX_PIXELS = 64_000_000


def allowed_types_label() -> str:
    """Human-readable list of accepted formats, for the rejection message."""
    return ", ".join(sorted(ext.lstrip(".").upper() for ext in ALLOWED_IMAGE_TYPES.values()))


def sniff_content_type(data: bytes) -> str | None:
    """Return the MIME type of ``data`` by its magic bytes, or None if unsupported.

    Deliberately narrow — only the :data:`ALLOWED_IMAGE_TYPES` formats are recognised,
    so anything else (an executable renamed to ``.jpg``, an HTML page, a zip) sniffs to
    None and is rejected.
    """
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"

    # WebP: "RIFF" <4-byte size> "WEBP".
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    return None


def decode_verified(data: bytes) -> tuple[int, int]:
    """Prove ``data`` really is a decodable image and return its ``(width, height)``.

    Raises ValueError on anything Pillow cannot open, on a truncated or corrupt file,
    and on a declared canvas above :data:`MAX_PIXELS`. The size is read from the header
    *before* the full decode, so a bomb is refused without ever being rasterised.
    """
    try:
        with Image.open(BytesIO(data)) as img:
            width, height = img.size

            if width * height > MAX_PIXELS:
                raise ValueError(f"image is {width}x{height}, over the {MAX_PIXELS} pixel cap")

            # verify() walks the whole file and catches truncation/corruption, but it
            # leaves the instance unusable — hence the size read above, and the reopen
            # in strip_metadata rather than reusing this handle.
            img.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"not a decodable image: {exc}") from exc

    return width, height


def strip_metadata(data: bytes, content_type: str) -> bytes:
    """Re-encode through Pillow so no metadata survives to disk (§12 item 4).

    Belt and braces: the Mac already strips EXIF by round-tripping through OpenCV, and
    the normalized render is synthesised so it never had any. This is the copy that
    holds if a future client forgets — GPS coordinates of a shooting spot are exactly
    the kind of thing that must not be one path-traversal away from leaking.

    Every branch re-encodes without growing the file, because the whole stack lives on
    one LUKS SSD whose backup job is still post-MVP:

    - JPEG: ``quality="keep"`` reuses the source quantization tables, so the metadata
      dies and the pixels do not. These originals are Phase-3 YOLO training data.
    - PNG: no ``optimize`` — it costs a zlib-9 filter search on every ingest, on a
      Raspberry Pi, to shave a few percent off an image that is already synthetic.
    - WebP: high-quality lossy. Pillow does not report whether the source was lossless,
      and forcing ``lossless=True`` would *inflate* a lossy phone WebP several times
      over — so the lossy branch is the one that cannot hurt. A lossless WebP original
      loses a little here; that is the accepted cost, and WebP is the rare path anyway
      (phones send JPEG, the normalized render is always PNG).
    """
    out = BytesIO()

    with Image.open(BytesIO(data)) as img:
        if content_type == "image/jpeg":
            img.save(out, format="JPEG", quality="keep")
        elif content_type == "image/png":
            img.save(out, format="PNG")
        else:
            img.save(out, format="WEBP", quality=95)

    return out.getvalue()


def store(base: Path, subdir: str, data: bytes, content_type: str) -> str:
    """Write ``data`` under a generated name; return its path relative to ``base``.

    ``content_type`` has already been validated by :func:`sniff_content_type`, and
    ``subdir`` is a literal from the caller — neither is ever client input.
    """
    rel = f"{subdir}/{uuid.uuid4().hex}{ALLOWED_IMAGE_TYPES[content_type]}"
    target = base / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

    return rel


def delete(base: Path, rel_path: str) -> None:
    """Remove a stored file; a missing one is not an error (rollback runs blind)."""
    (base / rel_path).unlink(missing_ok=True)
