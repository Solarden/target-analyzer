"""The wire contract shared by the Mac client and the Pi server.

Keeping these in one importable package means the two sides can never drift on
the shape of what crosses the network. See implementation.md §3.
"""

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Blast radius, not realism: every hit becomes a row inserted in one transaction.
MAX_HITS = 200

# The detectors that exist. A client that runs one must find its name here, or the
# payload it builds is refused.
Method = Literal["manual", "cv_blob", "vlm", "cv_blob_vlm", "yolo"]
# The hand-marked reading: the one a person confirmed, and the baseline every detector is
# measured against. Both sides order by it, so both sides read it from here.
GROUND_TRUTH_METHOD: Method = "manual"


# Refused rather than dropped: a client ahead of its server would otherwise have the new
# field discarded and the rest stored, for good, since re-sending a held photo is a 409.
class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Hit(_Wire):
    # A non-finite coordinate scores as a miss and then breaks the metrics JSON. The
    # range check against canon_size_px is the server's (§7).
    x_canon: float = Field(allow_inf_nan=False)
    y_canon: float = Field(allow_inf_nan=False)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class SessionMeta(_Wire):
    gun: str = Field(max_length=100)
    # Indexed with `gun` and filtered on, so a non-finite value matches nothing ever.
    distance_m: float = Field(gt=0, allow_inf_nan=False)
    notes: str = Field(default="", max_length=10_000)
    shot_at: date | None = None  # the shooting day; EXIF is stripped, so it can't be derived
    target_profile: str = Field(max_length=100)
    target_profile_version: int = Field(ge=1)


class ShipPayload(_Wire):
    schema_version: int = 1
    # The idempotency key, and a unique index: lowercase hex or nothing.
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canon_size_px: int = Field(gt=0)  # must equal the profile's (the frame contract)
    # Which detector produced these hits. One reading per method per photo, so a second
    # method on one photo is another interpretation rather than a replacement.
    method: Method = "manual"
    model: str | None = None
    params: dict = Field(default_factory=dict)
    session: SessionMeta
    hits: list[Hit] = Field(max_length=MAX_HITS)
    # Packaging time, tz-aware because a payload can queue across a DST shift. It has
    # no column: the server's own timestamps are authoritative, so this rides in params.
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
