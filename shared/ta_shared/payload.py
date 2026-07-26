"""The wire contract shared by the Mac client and the Pi server.

Keeping these in one importable package means the two sides can never drift on
the shape of what crosses the network. See internal_docs/implementation.md §3.
"""

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Hit(BaseModel):
    # allow_inf_nan=False: a non-finite coordinate has no place in the canonical frame,
    # and inf/NaN would sail through scoring as a "miss" and then break the JSON that
    # carries the metrics. The range check against canon_size_px is the server's (§7).
    x_canon: float = Field(allow_inf_nan=False)
    y_canon: float = Field(allow_inf_nan=False)
    confidence: float | None = None


class SessionMeta(BaseModel):
    gun: str
    distance_m: float
    notes: str = ""
    shot_at: date | None = None  # the shooting day; EXIF is stripped, so it can't be derived
    target_profile: str
    target_profile_version: int


class ShipPayload(BaseModel):
    schema_version: int = 1
    image_sha256: str  # identity + idempotency key: sha256 of the stripped original
    canon_size_px: int  # must equal the profile's canon_size_px (the frame contract)
    method: Literal["manual"] = "manual"  # Phase 2 adds cv_blob | vlm | yolo
    model: str | None = None
    params: dict = Field(default_factory=dict)
    session: SessionMeta
    hits: list[Hit]
    # When the client packaged this, UTC-aware — a payload can sit in the outbox
    # across a DST shift before it is ever read. No column of its own: the server's
    # own timestamps are authoritative, so ingest keeps this in
    # `interpretation.params` alongside the other client provenance.
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
