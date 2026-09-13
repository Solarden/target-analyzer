"""Assembling the payload and writing the folder that carries it.
See internal_docs/implementation.md §3 and §9.

Nothing here measures or computes anything — it puts what the other modules produced
into the shape the server validates.
"""

import os
import shutil
import tempfile
from pathlib import Path

from ta_client import __version__
from ta_client.register import Registration
from ta_shared.payload import Hit, SessionMeta, ShipPayload
from ta_shared.profile import TargetProfile


def build_payload(
    *,
    image_sha256: str,
    profile: TargetProfile,
    registration: Registration,
    hits: list[Hit],
    session: SessionMeta,
) -> ShipPayload:
    return ShipPayload(
        image_sha256=image_sha256,
        canon_size_px=profile.canon_size_px,
        method="manual",
        model=None,
        params=registration.params() | {"client_version": __version__},
        session=session,
        hits=hits,
    )


def write_session_dir(
    dest: Path, payload: ShipPayload, normalized_png: bytes, original_jpg: bytes
) -> None:
    """Write the three files of one session, appearing at ``dest`` all at once.

    Staged in a sibling directory and moved into place, so an interrupted run leaves
    either nothing or a complete folder — never half of one for the outbox to flush.
    """
    if dest.exists():
        raise FileExistsError(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}-", dir=dest.parent))

    try:
        (staging / "payload.json").write_text(payload.model_dump_json(), encoding="utf-8")
        (staging / "normalized.png").write_bytes(normalized_png)
        (staging / "original.jpg").write_bytes(original_jpg)
        os.replace(staging, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
