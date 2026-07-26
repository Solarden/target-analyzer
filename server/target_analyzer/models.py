"""SQLModel table definitions — the whole MVP schema.

Importing this module registers every table on SQLModel.metadata, which is what
Alembic autogenerate reads. See internal_docs/implementation.md §5. Decisions
baked in here:

- **Image and interpretation are decoupled.** One image carries many
  interpretations (``manual`` / ``cv_blob`` / ``vlm`` / ``yolo``), keyed
  ``UNIQUE(image_id, method)``. The MVP writes exactly one; Phase 2 adds a row
  per method and the Compare view lights up with no schema change.
- **Target profiles are immutable + versioned.** Re-measuring the physical
  target *inserts* ``version + 1``; it never UPDATEs. So re-scoring is
  reproducible and every interpretation pins the geometry it actually scored.
- **``interpretation`` caches the headline metrics** as columns (the Trend view
  reads them without recomputing) and keeps the full set in ``metrics``.
- Coordinates and metrics are honest floats — unlike money, a pixel has no
  minor unit.
- JSON columns are the generic SQLAlchemy ``JSON``, not ``JSONB``, so this one
  model set runs on both the Pi's Postgres and a local SQLite dev database.
"""

from datetime import UTC, date, datetime

from sqlalchemy import Column, ForeignKey, Index, Integer, UniqueConstraint
from sqlmodel import JSON, Field, SQLModel


def utc_now() -> datetime:
    """Current instant as a timezone-aware UTC datetime.

    Everything persisted is UTC; localizing is a presentation concern.

    Note the asymmetry: the timestamp columns are plain ``DateTime``, so the offset
    is dropped on write and a row reads back **naive** — on both dialects, which is
    the point (``timezone=True`` would hand back aware datetimes on Postgres and
    naive ones on SQLite, so dev and prod would diverge). Compare a stored timestamp
    against ``utc_now().replace(tzinfo=None)``, never against ``utc_now()`` itself.
    """
    return datetime.now(UTC)


class User(SQLModel, table=True):
    """A dashboard login. Single household, so no roles — see §8."""

    # Reserved word in Postgres; SQLAlchemy quotes it automatically. Hand-written
    # psql needs "user".
    __tablename__ = "user"

    id: int | None = Field(default=None, primary_key=True)
    name: str  # display name
    username: str = Field(unique=True, index=True)  # login handle
    password_hash: str
    is_active: bool = Field(default=True)  # deactivate without deleting
    created_at: datetime = Field(default_factory=utc_now)


class TargetProfile(SQLModel, table=True):
    """The geometry of one physical target, at one version (§4).

    Mirrors ``profiles/<name>.json`` (``ta_shared.profile.TargetProfile``) — the
    same shape, so :mod:`target_analyzer.scoring` scores from either without a
    conversion step. ``ring_radii_px`` is ascending, in canonical pixels, and
    ``len(ring_radii_px) == n_rings``; the innermost radius is ring ``n_rings``.

    ``target_diam_mm`` is the physical diameter of the **outer scoring ring**
    (not the card): it is what converts pixels to millimetres. NULL means mm
    metrics are simply not reported.

    ``board`` is the client's ArUco layout. The server never reads it; it lives
    here so "the target" stays one versioned, reproducible artifact.
    """

    __tablename__ = "target_profile"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_target_profile_name_version"),)

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)  # e.g. "issf_precision"
    version: int
    n_rings: int
    canon_size_px: int  # the canonical frame is this square (§1)
    ring_radii_px: list[int] = Field(sa_column=Column(JSON, nullable=False))
    target_diam_mm: float | None = Field(default=None)
    board: dict = Field(sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now)


class ShootingSession(SQLModel, table=True):
    """One string of shots at one target.

    Named ``ShootingSession`` (table ``session``) so it never shadows
    ``sqlmodel.Session`` in the modules that hold both.

    ``notes`` is free text the user edits later from the dashboard — it is the
    context that stops a bad day from reading as a regression.
    """

    __tablename__ = "session"
    # Trend filters by gun and distance together.
    __table_args__ = (Index("ix_session_gun_distance", "gun", "distance_m"),)

    id: int | None = Field(default=None, primary_key=True)
    target_profile_id: int = Field(foreign_key="target_profile.id", index=True)
    gun: str
    distance_m: float
    notes: str = Field(default="")
    shot_at: date | None = Field(default=None)  # the shooting day, user-entered (EXIF is stripped)
    created_at: datetime = Field(default_factory=utc_now)


class Image(SQLModel, table=True):
    """One photographed target: the EXIF-stripped original plus the warped
    canonical render (§7).

    ``sha256`` (of the stripped original bytes) is the identity of a photo and
    the idempotency key for ingest — a replayed outbox flush finds this row
    instead of creating a second session. The normalized image always exists (it
    is the dashboard's display surface); the original is opt-out
    (``TA_SHIP_ORIGINAL``), so ``original_path`` and the ``width``/``height`` that
    describe it are NULL together. The normalized render needs no dimensions of its
    own — it is ``canon_size_px`` square by construction.
    """

    __tablename__ = "image"

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(
        sa_column=Column(
            Integer, ForeignKey("session.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    sha256: str = Field(unique=True, index=True)
    original_path: str | None = Field(default=None)
    normalized_path: str
    canon_size_px: int  # what the client actually warped to; == the profile's (§1)
    width: int | None = Field(default=None)  # of the original, so NULL when it was not shipped
    height: int | None = Field(default=None)
    content_type: str  # sniffed from the bytes, never the client's declared type
    uploaded_at: datetime = Field(default_factory=utc_now)


class Interpretation(SQLModel, table=True):
    """One reading of one image by one method, with its scores (§5).

    ``method`` is a plain string, deliberately **not** a SQLModel enum: an enum
    becomes a native Postgres type, so Phase 2's ``cv_blob`` / ``vlm`` / ``yolo``
    would each need an ``ALTER TYPE``. The wire contract
    (:class:`ta_shared.payload.ShipPayload`) is the validating boundary; this
    column just stores what ran.

    ``target_profile_id`` pins the geometry version actually scored, so a later
    re-measure cannot retroactively change what this row means. The headline
    columns are a cache of :meth:`target_analyzer.scoring.ScoreResult.headline_columns`;
    ``metrics`` holds the full set (centroid, bias vector, sigmas, px variants).
    """

    __tablename__ = "interpretation"
    __table_args__ = (
        UniqueConstraint("image_id", "method", name="uq_interpretation_image_method"),
    )

    id: int | None = Field(default=None, primary_key=True)
    image_id: int = Field(
        sa_column=Column(
            Integer, ForeignKey("image.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    target_profile_id: int = Field(foreign_key="target_profile.id", index=True)
    method: str  # manual | cv_blob | vlm | yolo
    model: str | None = Field(default=None)  # e.g. "yolov8n-target-v3"; None for manual
    params: dict = Field(sa_column=Column(JSON, nullable=False))  # client version, H, residual, …
    created_at: datetime = Field(default_factory=utc_now)

    n_holes: int
    total_score: int
    avg_score: float | None = Field(default=None)  # None when there are no hits
    best_score: int | None = Field(default=None)
    extreme_spread_mm: float | None = Field(default=None)
    mean_radius_mm: float | None = Field(default=None)
    bias_mm: float | None = Field(default=None)
    metrics: dict = Field(sa_column=Column(JSON, nullable=False))


class Hole(SQLModel, table=True):
    """A single hit, in canonical pixels, with the ring the server scored it as.

    ``ring`` is 1..n_rings, or 0 for a miss. It is derived — recomputable from
    ``(x_canon, y_canon)`` and the interpretation's profile — but stored so the
    dashboard and Compare view read it without re-deriving.
    """

    __tablename__ = "hole"

    id: int | None = Field(default=None, primary_key=True)
    interpretation_id: int = Field(
        sa_column=Column(
            Integer, ForeignKey("interpretation.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    x_canon: float
    y_canon: float
    ring: int  # 0 = miss
    confidence: float | None = Field(default=None)  # None for manual clicks
