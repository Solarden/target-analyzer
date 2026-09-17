"""The outbox and the sender that drains it. See implementation.md §9.

Flush-first: every run drains the whole outbox, so a reconnect self-heals. The session
just clicked is not a special case — it is written into the outbox like any other and is
simply the newest folder in it.

The sender never deserializes a payload: ``payload.json`` goes on the wire as the bytes
on disk, verbatim, so no coordinate passes through here and the frame contract (§1)
cannot be breached by anything in this file.
"""

import argparse
import re
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

import httpx2

from ta_client.config import ENV_FILE, Settings, get_settings
from ta_client.package import write_session_dir
from ta_shared.payload import GROUND_TRUTH_METHOD, ShipPayload

# {epoch}-{sha256[:12]}-{method}, and the filter that keeps a half-written staging dotfile
# out of the outbox — write_session_dir stages inside it until os.replace renames it.
NAME = re.compile(r"\d+-[0-9a-f]{12}(-[a-z0-9_]+)?")
# 409 is a success here: the server already holds the photo, so the folder is finished.
# is_success and raise_for_status() disagree, and would stall the outbox on every replay.
DONE = (201, 409)
# Separate from the upload timeout, so an unreachable server costs five seconds per
# folder rather than sixty.
CONNECT_TIMEOUT = 5.0
LOCAL = ("localhost", "127.0.0.1", "::1")


class ShipError(RuntimeError):
    """The server refused this folder for a reason retrying will not fix."""


def check_config(settings: Settings) -> None:
    """Refuse a run that cannot possibly ship, before any clicking is done."""
    if not settings.ingest_token.get_secret_value():
        raise ShipError(
            "TA_INGEST_TOKEN is not set — mint one with "
            f"`python -m target_analyzer.create_token` on the server, and put it in {ENV_FILE}"
        )

    url = urlparse(settings.server_url)

    # httpx raises UnsupportedProtocol or connects to nothing, and the first of those is a
    # RequestError — so without this the outbox reads a permanent typo as a brief outage.
    if url.scheme not in ("http", "https") or not url.hostname:
        raise ShipError(
            f"TA_SERVER_URL must be a full http:// or https:// URL — got {settings.server_url!r}"
        )

    if url.scheme == "http" and url.hostname not in LOCAL:
        print(
            f"warning: {url.hostname} is plain http, so the bearer token crosses the "
            "network in the clear",
            file=sys.stderr,
        )


def enqueue(
    outbox: Path, payload: ShipPayload, normalized_png: bytes, original_jpg: bytes | None
) -> Path:
    """Write one session into the outbox and return its folder."""
    # Epoch first so the outbox keeps the order the sessions were shot in, then the photo
    # and the method, because a reading is superseded only by the same method's.
    folder = outbox / f"{int(time.time())}-{payload.image_sha256[:12]}-{payload.method}"
    write_session_dir(folder, payload, normalized_png, original_jpg)

    return folder


class _Parts(NamedTuple):
    epoch: int
    photo: str
    method: str


def _parts(folder: Path) -> _Parts:
    """A session folder's name, read back.

    A name without a method reads as the ground truth, so such a folder ships rather than
    sitting in the outbox unnoticed.
    """
    epoch, photo, *method = folder.name.split("-", 2)

    return _Parts(int(epoch), photo, method[0] if method else GROUND_TRUTH_METHOD)


def _order(folder: Path) -> tuple[int, bool]:
    """Oldest first, then the ground truth ahead of a detector sharing its second.

    The epoch sorts as an integer: it is unpadded, so "10-" must not land before "2-".

    One run enqueues both readings of a photo, and a detector's reading alone on the
    server is a session scored by nobody.
    """
    parts = _parts(folder)

    return parts.epoch, parts.method != GROUND_TRUTH_METHOD


def pending(outbox: Path) -> list[Path]:
    """The outbox, oldest first, with superseded markings dropped.

    Superseded means the same photo read by the same method. Two methods of one photo are
    two readings to compare, and dropping either would be a silent loss.
    """
    if not outbox.is_dir():
        return []

    folders = sorted(
        (p for p in outbox.iterdir() if p.is_dir() and NAME.fullmatch(p.name)), key=_order
    )
    # The server keeps whichever marking reaches it first, so shipping the older of two
    # would make the stale one permanent and 409 the correction that replaced it.
    keep = set({(_parts(p).photo, _parts(p).method): p for p in folders}.values())

    for folder in folders:
        if folder not in keep:
            print(f"{folder.name}: superseded by a later reading of the same photo")
            shutil.rmtree(folder)

    return [p for p in folders if p in keep]


def build_request(folder: Path, settings: Settings) -> httpx2.Request:
    """One session folder as the multipart of §3. Separate so it can be read without a server."""
    files = {
        "normalized": ("normalized.png", (folder / "normalized.png").read_bytes(), "image/png")
    }
    original = folder / "original.jpg"

    # Its presence is the whole TA_SHIP_ORIGINAL gate — package.py made that call at write
    # time, so nothing here reads the setting.
    if original.is_file():
        files["original"] = ("original.jpg", original.read_bytes(), "image/jpeg")

    return httpx2.Request(
        "POST",
        f"{settings.server_url.rstrip('/')}/api/ingest",
        # A form field, not a file part: the endpoint declares payload as Form(str).
        data={"payload": (folder / "payload.json").read_text(encoding="utf-8")},
        files=files,
        headers={"Authorization": f"Bearer {settings.ingest_token.get_secret_value()}"},
    )


def post(client: httpx2.Client, folder: Path, settings: Settings) -> httpx2.Response:
    return client.send(build_request(folder, settings))


Sender = Callable[[httpx2.Client, Path, Settings], httpx2.Response]


def flush(outbox: Path, *, settings: Settings | None = None, send: Sender = post) -> int:
    """Drain the outbox oldest first. Returns how many folders are still in it."""
    settings = settings or get_settings()
    folders = pending(outbox)
    timeout = httpx2.Timeout(settings.ship_timeout, connect=CONNECT_TIMEOUT)

    # One client for the whole drain, so a backlog is one connection rather than one
    # handshake per session.
    with httpx2.Client(timeout=timeout) as client:
        return _drain(client, folders, settings, send)


def _drain(client: httpx2.Client, folders: list[Path], settings: Settings, send: Sender) -> int:
    for index, folder in enumerate(folders):
        remaining = len(folders) - index

        try:
            response = send(client, folder, settings)
        except httpx2.RequestError as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)

            return remaining

        if response.status_code in DONE:
            shutil.rmtree(folder)
            # A 409 on a re-marking means these hits were not applied, not that they were.
            note = (
                "sent"
                if response.status_code == 201
                else "already held — this marking was not applied"
            )
            print(f"{folder.name}: {note}")
            continue

        # Not followed on purpose — that is what keeps the token on a single host — so a
        # proxy redirecting http to https surfaces here instead of as a working upload.
        if 300 <= response.status_code < 400:
            raise ShipError(
                f"{settings.server_url} answered {response.status_code}: it redirects, and "
                "redirects are not followed. Check TA_SERVER_URL's scheme — the outbox is fine."
            )

        if response.status_code >= 500:
            print(f"server error {response.status_code}", file=sys.stderr)

            return remaining

        # ponytail: a folder the server will never accept blocks everything behind it. A
        # failed/ quarantine dir is the upgrade if that ever happens twice.
        raise ShipError(
            f"{folder} was refused with {response.status_code}: {response.text[:500]}\n"
            "Shipping stops here — fix the cause, or delete that folder to unblock the rest."
        )

    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ta_client.ship", description="Drain the outbox.")
    settings = get_settings()
    parser.add_argument(
        "--outbox", type=Path, default=settings.outbox_path, metavar="DIR", help="the outbox"
    )
    args = parser.parse_args(argv)

    if not settings.server_url:
        parser.error(f"TA_SERVER_URL is not set — see the README and {ENV_FILE}")

    try:
        check_config(settings)
        queued = flush(args.outbox, settings=settings)
    except ShipError as exc:
        print(exc, file=sys.stderr)

        return 1

    print(f"{queued} queued" if queued else "outbox empty")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
