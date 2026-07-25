"""The wire contract shared by the Mac client and the Pi server.

Keeping these in one importable package means the two sides can never drift on
the shape of what crosses the network. See internal_docs/implementation.md §3.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Hit(BaseModel):
    x_canon: float
    y_canon: float
    confidence: float | None = None  # always None for manual clicks


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
    created_at: datetime = Field(default_factory=datetime.now)
