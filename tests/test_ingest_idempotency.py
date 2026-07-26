"""POST /api/ingest — idempotency and the guardrails around it (implementation.md §7).

The idempotency cases are the reason this file exists: the Mac's outbox replays on
reconnect, so a resent flush must find the existing rows rather than mint a second
session. Everything after them is a boundary that would otherwise only be discovered in
production, by mis-scored data.
"""

import hashlib
from io import BytesIO

import pytest
from fastapi import HTTPException, Response, status
from PIL import Image as PILImage
from sqlmodel import Session, func, select
from tests.conftest import make_jpeg, make_payload, make_png

from target_analyzer.api.endpoints.ingest import _ingest_onto_existing_image
from target_analyzer.config import get_settings
from target_analyzer.models import Hole, Image, Interpretation, ShootingSession


def post(client, auth, payload, png, *, jpeg=None):
    files = {"normalized": ("n.png", png, "image/png")}

    if jpeg is not None:
        files["original"] = ("o.jpg", jpeg, "image/jpeg")

    return client.post(
        "/api/ingest",
        headers=auth,
        data={"payload": payload.model_dump_json()},
        files=files,
    )


def count(session: Session, model) -> int:
    return session.exec(select(func.count()).select_from(model)).one()


# --- the three idempotency cases -------------------------------------------------


def test_a_new_photo_creates_a_session_image_interpretation_and_holes(
    client, auth, profile, db_session, png_bytes
):
    jpeg = make_jpeg()
    response = post(client, auth, make_payload(jpeg), png_bytes, jpeg=jpeg)
    body = response.json()

    assert response.status_code == status.HTTP_201_CREATED
    assert body["duplicate"] is False
    assert body["sha256"] == hashlib.sha256(jpeg).hexdigest()
    assert (body["n_holes"], body["total_score"]) == (2, 20)
    assert body["metrics"]["bias_direction"] == "right"
    assert (count(db_session, ShootingSession), count(db_session, Image)) == (1, 1)
    assert (count(db_session, Interpretation), count(db_session, Hole)) == (1, 2)


def test_replaying_the_same_flush_returns_the_existing_rows(
    client, auth, profile, db_session, png_bytes
):
    """The outbox case: same photo, same method, twice. 409 and nothing new."""
    jpeg = make_jpeg()
    first = post(client, auth, make_payload(jpeg), png_bytes, jpeg=jpeg).json()

    replay = post(client, auth, make_payload(jpeg), png_bytes, jpeg=jpeg)

    assert replay.status_code == status.HTTP_409_CONFLICT
    assert replay.json()["duplicate"] is True
    assert replay.json()["interpretation_id"] == first["interpretation_id"]
    assert replay.json()["session_id"] == first["session_id"]
    assert (count(db_session, ShootingSession), count(db_session, Interpretation)) == (1, 1)


def test_a_second_method_attaches_to_the_same_image(client, auth, profile, db_session, png_bytes):
    """The Phase-2 shape: one image, many readings, no schema change (§13).

    Driven below HTTP on purpose. ``method`` is ``Literal["manual"]`` on the wire, so
    the contract — correctly — refuses to let a ``cv_blob`` payload through the endpoint
    today. What is worth proving now is the branch behind it: a second reading of a known
    photo must attach to the *existing* image and session rather than mint new ones, and
    must not write a second copy of the files. That is what UNIQUE(image_id, method)
    exists for and what the Compare view will render side by side.
    """
    jpeg = make_jpeg()
    first = post(client, auth, make_payload(jpeg), png_bytes, jpeg=jpeg).json()
    on_disk = {p for p in get_settings().data_path.rglob("*") if p.is_file()}

    ship = make_payload(jpeg)
    ship.method = "cv_blob"  # no validate_assignment, so this bypasses the wire Literal
    image = db_session.exec(select(Image)).one()
    body = _ingest_onto_existing_image(db_session, Response(), image, profile, ship)

    assert body["duplicate"] is False
    assert body["image_id"] == first["image_id"]
    assert body["session_id"] == first["session_id"]
    assert body["interpretation_id"] != first["interpretation_id"]
    assert (count(db_session, Image), count(db_session, Interpretation)) == (1, 2)
    assert {p for p in get_settings().data_path.rglob("*") if p.is_file()} == on_disk


def test_a_second_reading_in_a_different_frame_is_rejected(
    client, auth, profile, db_session, png_bytes
):
    """The attach path has no render to measure, so the stored frame is the only witness.

    A payload naming a profile version with a different canonical square would otherwise
    pass the payload-vs-profile check and then score hits against geometry the image on
    disk was never warped to.
    """
    jpeg = make_jpeg()
    post(client, auth, make_payload(jpeg), png_bytes, jpeg=jpeg)

    ship = make_payload(jpeg)
    ship.method = "cv_blob"
    ship.canon_size_px = 800
    profile.canon_size_px = 800  # a re-measured profile version, consistent with itself
    image = db_session.exec(select(Image)).one()

    with pytest.raises(HTTPException) as raised:
        _ingest_onto_existing_image(db_session, Response(), image, profile, ship)

    assert raised.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "warped to" in raised.value.detail


# --- auth ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "Bearer "},
    ],
)
def test_bad_or_missing_credentials_are_401(client, profile, png_bytes, headers):
    """A missing header gets the same 401 as a wrong one — never FastAPI's default 403."""
    response = post(client, headers, make_payload(make_jpeg()), png_bytes)

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_an_unconfigured_server_is_503_not_401(client, auth, profile, png_bytes, monkeypatch):
    """The caller's token is fine; the server is unfinished. A 401 would send the Mac
    into a retry loop against a box that can never accept it."""
    monkeypatch.setattr(get_settings(), "ingest_token_hash", None)

    response = post(client, auth, make_payload(make_jpeg()), png_bytes)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


# --- the frame contract (§1) -----------------------------------------------------


def test_a_declared_frame_size_mismatch_is_rejected(client, auth, profile, png_bytes):
    """Half one: warping to a different square than the profile mis-scores every ring."""
    response = post(client, auth, make_payload(make_jpeg(), canon_size_px=800), png_bytes)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "canon_size_px" in response.json()["detail"]


def test_a_normalized_render_of_the_wrong_size_is_rejected(client, auth, profile):
    """Half two, the one that actually holds: the payload can claim 1000 all it likes."""
    response = post(client, auth, make_payload(make_jpeg()), make_png(800))

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "800x800" in response.json()["detail"]


@pytest.mark.parametrize(
    "hit", [{"x_canon": -1.0, "y_canon": 500.0}, {"x_canon": 500.0, "y_canon": 1001.0}]
)
def test_a_hit_outside_the_canonical_frame_is_rejected(client, auth, profile, png_bytes, hit):
    payload = make_payload(make_jpeg())
    raw = payload.model_dump_json().replace(
        '"x_canon":500.0,"y_canon":500.0', f'"x_canon":{hit["x_canon"]},"y_canon":{hit["y_canon"]}'
    )
    response = client.post(
        "/api/ingest",
        headers=auth,
        data={"payload": raw},
        files={"normalized": ("n.png", png_bytes, "image/png")},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "canonical frame" in response.json()["detail"]


# --- identity, profile, schema ---------------------------------------------------


def test_a_hash_that_does_not_match_the_original_is_rejected(client, auth, profile, png_bytes):
    """With the original in hand the claimed identity is verifiable, so it is verified."""
    payload = make_payload(make_jpeg())  # hash of one image...
    response = post(client, auth, payload, png_bytes, jpeg=make_jpeg(size=48))  # ...body of another

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "does not match" in response.json()["detail"]


def test_an_unknown_profile_is_404(client, auth, db_session, png_bytes):
    """No profile seeded at all — the deploy step that P6 owns."""
    response = post(client, auth, make_payload(make_jpeg()), png_bytes)

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "seed_profile" in response.json()["detail"]


def test_a_future_schema_version_is_rejected(client, auth, profile, png_bytes):
    """Fail loud rather than mis-read a payload written to a contract we don't speak."""
    payload = make_payload(make_jpeg())
    raw = payload.model_dump_json().replace('"schema_version":1', '"schema_version":2')
    response = client.post(
        "/api/ingest",
        headers=auth,
        data={"payload": raw},
        files={"normalized": ("n.png", png_bytes, "image/png")},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "schema_version" in response.json()["detail"]


# --- upload hardening ------------------------------------------------------------


def test_a_non_image_upload_is_rejected(client, auth, profile):
    response = post(client, auth, make_payload(make_jpeg()), b"MZ\x90\x00 this is an executable")

    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


def test_the_normalized_part_must_be_png(client, auth, profile):
    """It is the dashboard's overlay surface and is rendered, not photographed."""
    response = post(client, auth, make_payload(make_jpeg()), make_jpeg(1000))

    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    assert "image/png" in response.json()["detail"]


def test_an_oversized_upload_is_rejected(client, auth, profile, png_bytes, monkeypatch):
    monkeypatch.setattr(get_settings(), "attachment_max_bytes", 16)

    response = post(client, auth, make_payload(make_jpeg()), png_bytes)

    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE


# --- what lands on disk ----------------------------------------------------------


def test_exif_is_stripped_from_the_stored_original(client, auth, profile, db_session, png_bytes):
    """Re-stripped server-side even though the Mac already did it (§12 item 4)."""
    jpeg = make_jpeg(with_exif=True)
    assert PILImage.open(BytesIO(jpeg)).getexif()

    post(client, auth, make_payload(jpeg), png_bytes, jpeg=jpeg)

    image = db_session.exec(select(Image)).one()
    stored = get_settings().data_path / image.original_path

    assert not PILImage.open(stored).getexif()


def test_a_skipped_original_leaves_its_columns_null(client, auth, profile, db_session, png_bytes):
    """TA_SHIP_ORIGINAL is opt-out on the Mac; the normalized render always exists."""
    post(client, auth, make_payload(make_jpeg()), png_bytes)

    image = db_session.exec(select(Image)).one()

    assert (image.original_path, image.width, image.height) == (None, None, None)
    assert (get_settings().data_path / image.normalized_path).exists()
    assert image.content_type == "image/png"


def test_a_failed_commit_leaves_no_orphan_files(
    client, auth, profile, db_session, png_bytes, monkeypatch
):
    """Files are written before the transaction, so a DB failure must unlink them."""
    from target_analyzer.api.endpoints import ingest as ingest_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("commit exploded")

    monkeypatch.setattr(ingest_module, "_add_interpretation", boom)
    data_path = get_settings().data_path
    before = {p for p in data_path.rglob("*") if p.is_file()}

    with pytest.raises(RuntimeError):
        post(client, auth, make_payload(make_jpeg()), png_bytes, jpeg=make_jpeg())

    assert {p for p in data_path.rglob("*") if p.is_file()} == before
    assert count(db_session, Image) == 0
