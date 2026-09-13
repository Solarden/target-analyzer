"""The wire contract shared by the Mac client and the Pi server.

Keeping these in one importable package means the two sides can never drift on
the shape of what crosses the network. See internal_docs/implementation.md §3.
"""

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, Field

# A shooting string is 5-60 shots. The bound is not about realism but about blast
# radius: every hit becomes a `hole` row inserted in one transaction.
MAX_HITS = 200


class Hit(BaseModel):
    # allow_inf_nan=False: a non-finite coordinate has no place in the canonical frame,
    # and inf/NaN would sail through scoring as a "miss" and then break the JSON that
    # carries the metrics. The range check against canon_size_px is the server's (§7).
    x_canon: float = Field(allow_inf_nan=False)
    y_canon: float = Field(allow_inf_nan=False)
    # Same reasoning one level down: this lands in `hole.confidence` and is rendered.
    # None for manual clicks; Phase-2 detectors are what actually populate it.
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class SessionMeta(BaseModel):
    gun: str = Field(max_length=100)
    # Indexed together with `gun` and drives the Trend filter, so a non-finite value
    # would match no filter ever — the row would just quietly vanish from the dashboard.
    distance_m: float = Field(gt=0, allow_inf_nan=False)
    notes: str = Field(default="", max_length=10_000)
    shot_at: date | None = None  # the shooting day; EXIF is stripped, so it can't be derived
    target_profile: str = Field(max_length=100)
    target_profile_version: int = Field(ge=1)


class ShipPayload(BaseModel):
    schema_version: int = 1
    # Identity + idempotency key: sha256 of the stripped original. Pinned to lowercase
    # hex because it *is* the unique index — anything else poisons it.
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canon_size_px: int = Field(gt=0)  # must equal the profile's (the frame contract)
    method: Literal["manual"] = "manual"  # Phase 2 adds cv_blob | vlm | yolo
    model: str | None = None
    params: dict = Field(default_factory=dict)
    session: SessionMeta
    hits: list[Hit] = Field(max_length=MAX_HITS)
    # When the client packaged this, UTC-aware — a payload can sit in the outbox
    # across a DST shift before it is ever read. No column of its own: the server's
    # own timestamps are authoritative, so ingest keeps this in
    # `interpretation.params` alongside the other client provenance.
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
