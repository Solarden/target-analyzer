"""``POST /api/ingest`` — the only way data enters the system (implementation.md §7).

The Mac ships canonical hit coordinates plus the images; this scores them and persists
the result. Three properties are load-bearing and each is guarded explicitly below:

- **The frame contract (§1).** The client's warp and the server's ring radii must live
  in the same canonical square. A silent mismatch mis-scores every boundary shot and
  inverts the bias read, so it is rejected twice: once on the declared
  ``canon_size_px``, and once on the actual pixel size of the normalized render — the
  second is the one a buggy client cannot talk its way past.
- **Idempotency by sha256.** The Mac's outbox replays on reconnect. A replayed flush
  must find the existing row, never create a second session.
- **No orphans.** Files are written before the transaction and unlinked if it fails, so
  the SSD never accumulates blobs no row points at.
"""

import hashlib
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from sqlmodel import Session, select

from ta_shared.payload import ShipPayload
from target_analyzer import storage
from target_analyzer.api.deps import DbSession
from target_analyzer.bearer import require_machine
from target_analyzer.config import get_settings
from target_analyzer.models import Hole, Image, Interpretation, ShootingSession, TargetProfile
from target_analyzer.scoring import compute_metrics

router = APIRouter(prefix="/api", tags=["ingest"], dependencies=[Depends(require_machine)])

# The wire contract's version. Bumped only by a breaking change to ShipPayload; the
# check exists so a newer client fails loudly against an older Pi instead of having its
# payload quietly misread.
SUPPORTED_SCHEMA_VERSION = 1


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


async def _read_capped(upload: UploadFile, label: str) -> bytes:
    """Read one part, refusing anything over the configured cap."""
    max_bytes = get_settings().attachment_max_bytes
    too_large = HTTPException(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        detail=f"{label} is larger than the {max_bytes} byte limit",
    )

    # Cheap pre-read guard on the declared part size, so an oversized body is not read
    # fully into memory before being rejected; len(data) below is authoritative.
    if upload.size is not None and upload.size > max_bytes:
        raise too_large

    data = await upload.read()

    if not data:
        raise _bad_request(f"{label} is empty")

    if len(data) > max_bytes:
        raise too_large

    return data


def _sniff(data: bytes, label: str, *, expected: str | None = None) -> str:
    """Decide a part's type from its bytes and check it against the allowlist."""
    content_type = storage.sniff_content_type(data)

    if content_type is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{label} is not one of: {storage.allowed_types_label()}",
        )

    if expected is not None and content_type != expected:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{label} must be {expected}, got {content_type}",
        )

    return content_type


def _decode(data: bytes, label: str) -> tuple[int, int]:
    """Decode-verify one part, turning storage's ValueError into a 400."""
    try:
        return storage.decode_verified(data)
    except ValueError as exc:
        raise _bad_request(f"{label}: {exc}") from exc


def _resolve_profile(session: Session, ship: ShipPayload) -> TargetProfile:
    profile = session.exec(
        select(TargetProfile).where(
            TargetProfile.name == ship.session.target_profile,
            TargetProfile.version == ship.session.target_profile_version,
        )
    ).first()

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"unknown target profile {ship.session.target_profile} "
                f"v{ship.session.target_profile_version} — seed it with "
                "`python -m target_analyzer.seed_profile`"
            ),
        )

    # The frame contract, half one. Half two is the pixel check in the create path: this
    # only verifies what the client *says* it warped to.
    if ship.canon_size_px != profile.canon_size_px:
        raise _bad_request(
            f"canon_size_px {ship.canon_size_px} does not match profile "
            f"{profile.name} v{profile.version} ({profile.canon_size_px})"
        )

    return profile


def _add_interpretation(
    session: Session, image_id: int, profile: TargetProfile, ship: ShipPayload
) -> Interpretation:
    """Score the hits and stage the interpretation + its holes. Does not commit."""
    # The DB row satisfies the RingGeometry protocol structurally, so it is scored
    # directly — no conversion to the JSON profile, and no second source of geometry.
    result = compute_metrics([(hit.x_canon, hit.y_canon) for hit in ship.hits], profile)

    interpretation = Interpretation(
        image_id=image_id,
        target_profile_id=profile.id,
        method=ship.method,
        model=ship.model,
        # created_at has no column of its own (see payload.py) — it is the client's
        # packaging time, provenance rather than truth, so it rides along here.
        params={**ship.params, "created_at": ship.created_at.isoformat()},
        metrics=result.metrics_jsonb(),
        **result.headline_columns(),
    )
    session.add(interpretation)
    session.flush()

    for hit, ring in zip(ship.hits, result.rings, strict=True):
        session.add(
            Hole(
                interpretation_id=interpretation.id,
                x_canon=hit.x_canon,
                y_canon=hit.y_canon,
                ring=ring,
                confidence=hit.confidence,
            )
        )

    return interpretation


def _body(
    session_id: int, image: Image, interpretation: Interpretation, *, duplicate: bool
) -> dict:
    return {
        "session_id": session_id,
        "image_id": image.id,
        "interpretation_id": interpretation.id,
        "sha256": image.sha256,
        "duplicate": duplicate,
        "n_holes": interpretation.n_holes,
        "total_score": interpretation.total_score,
        "metrics": interpretation.metrics,
    }


@router.post("/ingest")
async def ingest(
    session: DbSession,
    response: Response,
    payload: Annotated[str, Form()],
    normalized: Annotated[UploadFile, File()],
    original: Annotated[UploadFile | None, File()] = None,
) -> dict:
    ship = ShipPayload.model_validate_json(payload)

    if ship.schema_version != SUPPORTED_SCHEMA_VERSION:
        raise _bad_request(
            f"payload schema_version {ship.schema_version} is not supported "
            f"(this server speaks {SUPPORTED_SCHEMA_VERSION})"
        )

    profile = _resolve_profile(session, ship)

    if any(
        not (0 <= coord <= ship.canon_size_px)
        for hit in ship.hits
        for coord in (hit.x_canon, hit.y_canon)
    ):
        raise _bad_request(f"every hit must lie within the 0..{ship.canon_size_px} canonical frame")

    normalized_bytes = await _read_capped(normalized, "normalized")
    original_bytes = await _read_capped(original, "original") if original is not None else None

    # Identity. With the original in hand the client's hash is verifiable, so verify it;
    # shipping the original is optional (§3), and without it the client's hash is all
    # there is. Either way this is the hash of the bytes the *client* stripped — the copy
    # written to disk is re-encoded below and deliberately will not match it.
    if original_bytes is not None:
        recomputed = hashlib.sha256(original_bytes).hexdigest()

        if recomputed != ship.image_sha256:
            raise _bad_request(
                f"image_sha256 {ship.image_sha256} does not match the uploaded original "
                f"({recomputed})"
            )

    existing = session.exec(select(Image).where(Image.sha256 == ship.image_sha256)).first()

    if existing is not None:
        return _ingest_onto_existing_image(session, response, existing, profile, ship)

    return _ingest_new_image(session, response, profile, ship, normalized_bytes, original_bytes)


def _ingest_onto_existing_image(
    session: Session,
    response: Response,
    image: Image,
    profile: TargetProfile,
    ship: ShipPayload,
) -> dict:
    """This photo is already known: either a replay, or a second method reading it."""
    # The frame contract on the one path that cannot pixel-check it. There is no render
    # to measure here — the images on disk were warped when the image row was created —
    # so the stored canon_size_px is the only witness. Without this, a payload naming a
    # *different* profile version (one whose canonical square is a different size) would
    # pass the payload-vs-profile check above and then score hits against geometry the
    # image on disk was never warped to.
    if ship.canon_size_px != image.canon_size_px:
        raise _bad_request(
            f"canon_size_px {ship.canon_size_px} does not match the frame this image was "
            f"warped to ({image.canon_size_px})"
        )

    duplicate = session.exec(
        select(Interpretation).where(
            Interpretation.image_id == image.id, Interpretation.method == ship.method
        )
    ).first()

    # A replayed outbox flush. Idempotent success: nothing is created, no file is
    # written, and the client gets back the ids it would have got the first time.
    if duplicate is not None:
        response.status_code = status.HTTP_409_CONFLICT

        return _body(image.session_id, image, duplicate, duplicate=True)

    # Same photo, a method that has not read it yet — the Phase-2 path. Unreachable
    # while `method` is Literal["manual"], but it is what UNIQUE(image_id, method)
    # exists for, and it is the row the Compare view will render side by side. No new
    # files: the images on disk belong to the image row, not to one reading of it.
    interpretation = _add_interpretation(session, image.id, profile, ship)
    session.commit()
    session.refresh(interpretation)
    response.status_code = status.HTTP_201_CREATED

    return _body(image.session_id, image, interpretation, duplicate=False)


def _ingest_new_image(
    session: Session,
    response: Response,
    profile: TargetProfile,
    ship: ShipPayload,
    normalized_bytes: bytes,
    original_bytes: bytes | None,
) -> dict:
    normalized_type = _sniff(normalized_bytes, "normalized", expected="image/png")
    width, height = _decode(normalized_bytes, "normalized")

    # The frame contract, half two: the render itself must be the canonical square.
    if (width, height) != (profile.canon_size_px, profile.canon_size_px):
        raise _bad_request(
            f"normalized image is {width}x{height}, expected "
            f"{profile.canon_size_px}x{profile.canon_size_px}"
        )

    original_type = None
    original_size = None

    if original_bytes is not None:
        original_type = _sniff(original_bytes, "original")
        original_size = _decode(original_bytes, "original")

    data_path = get_settings().data_path
    written: list[str] = []

    try:
        normalized_path = storage.store(
            data_path,
            "normalized",
            storage.strip_metadata(normalized_bytes, normalized_type),
            normalized_type,
        )
        written.append(normalized_path)

        original_path = None

        if original_bytes is not None:
            original_path = storage.store(
                data_path,
                "originals",
                storage.strip_metadata(original_bytes, original_type),
                original_type,
            )
            written.append(original_path)

        shooting_session = ShootingSession(
            target_profile_id=profile.id,
            gun=ship.session.gun,
            distance_m=ship.session.distance_m,
            notes=ship.session.notes,
            shot_at=ship.session.shot_at,
        )
        session.add(shooting_session)
        session.flush()

        image = Image(
            session_id=shooting_session.id,
            sha256=ship.image_sha256,
            original_path=original_path,
            normalized_path=normalized_path,
            canon_size_px=ship.canon_size_px,
            width=original_size[0] if original_size else None,
            height=original_size[1] if original_size else None,
            # Describes the original when there is one; otherwise the normalized render,
            # which is then the only image on disk. The column is NOT NULL either way.
            content_type=original_type or normalized_type,
        )
        session.add(image)
        session.flush()

        interpretation = _add_interpretation(session, image.id, profile, ship)
        # ponytail: a concurrent POST of the same sha256 raises IntegrityError here and
        # 500s. The shipper is one Mac flushing its outbox in FIFO order, so there is no
        # second writer; if that ever changes, catch it and re-read as a duplicate.
        session.commit()
    except Exception:
        session.rollback()

        for path in written:
            storage.delete(data_path, path)
        raise

    session.refresh(image)
    session.refresh(interpretation)
    response.status_code = status.HTTP_201_CREATED

    return _body(image.session_id, image, interpretation, duplicate=False)
