"""The corpus export: what the server holds, back in the shape the client reads."""

import hashlib
import json
from datetime import UTC, datetime

from fastapi import status

from ta_shared.payload import Hit
from target_analyzer.config import get_settings
from target_analyzer.export_corpus import export
from tests.conftest import make_jpeg, make_payload, post_ingest


def _ingest(client, auth, png_bytes, *, size=64, **overrides):
    jpeg = make_jpeg(size)
    response = post_ingest(client, auth, make_payload(jpeg, **overrides), png_bytes, jpeg=jpeg)

    assert response.status_code == status.HTTP_201_CREATED

    return hashlib.sha256(jpeg).hexdigest()


def test_export_round_trips_a_marked_reading(
    client, auth, profile, png_bytes, db_session, tmp_path
):
    hits = [Hit(x_canon=500, y_canon=500), Hit(x_canon=511.5, y_canon=498.25)]
    digest = _ingest(client, auth, png_bytes, hits=hits)

    assert export(db_session, get_settings().data_path, tmp_path) == 1

    folder = next(tmp_path.iterdir())
    payload = json.loads((folder / "payload.json").read_text())

    assert folder.name.endswith(f"-{digest[:12]}-manual")
    assert payload["image_sha256"] == digest
    assert payload["session"]["target_profile"] == profile.name
    assert payload["session"]["target_profile_version"] == profile.version
    assert [(h["x_canon"], h["y_canon"]) for h in payload["hits"]] == [
        (h.x_canon, h.y_canon) for h in hits
    ]
    assert (folder / "normalized.png").read_bytes() == png_bytes


def test_export_leaves_a_detector_reading_behind(
    client, auth, profile, png_bytes, db_session, tmp_path
):
    """A proposal is not ground truth, so it must never reach the corpus."""
    jpeg = make_jpeg()
    post_ingest(client, auth, make_payload(jpeg), png_bytes, jpeg=jpeg)
    second = post_ingest(client, auth, make_payload(jpeg, method="cv_blob"), png_bytes)

    assert second.status_code in (status.HTTP_201_CREATED, status.HTTP_200_OK), second.text
    assert export(db_session, get_settings().data_path, tmp_path) == 1

    methods = {json.loads((f / "payload.json").read_text())["method"] for f in tmp_path.iterdir()}

    assert methods == {"manual"}


def test_export_puts_the_utc_zone_back(client, auth, profile, png_bytes, db_session, tmp_path):
    """Neither backend stores an offset, and a naive instant reports a local-time epoch."""
    _ingest(client, auth, png_bytes)
    export(db_session, get_settings().data_path, tmp_path)

    folder = next(tmp_path.iterdir())
    payload = json.loads((folder / "payload.json").read_text())
    created = datetime.fromisoformat(payload["created_at"])
    epoch = int(folder.name.split("-")[0])

    assert created.tzinfo is not None, payload["created_at"]
    assert abs(epoch - datetime.now(UTC).timestamp()) < 3600, folder.name


def test_export_keeps_each_photos_holes_apart(
    client, auth, profile, png_bytes, db_session, tmp_path
):
    one = _ingest(client, auth, png_bytes, size=64, hits=[Hit(x_canon=100, y_canon=100)])
    two = _ingest(
        client,
        auth,
        png_bytes,
        size=65,
        hits=[Hit(x_canon=900, y_canon=900), Hit(x_canon=901, y_canon=902)],
    )
    export(db_session, get_settings().data_path, tmp_path)

    exported = {}

    for folder in tmp_path.iterdir():
        payload = json.loads((folder / "payload.json").read_text())
        exported[payload["image_sha256"]] = [(h["x_canon"], h["y_canon"]) for h in payload["hits"]]

    assert exported == {one: [(100, 100)], two: [(900, 900), (901, 902)]}


def test_export_carries_who_shot_it(client, auth, profile, png_bytes, db_session, tmp_path):
    session = make_payload(make_jpeg()).session.model_copy(update={"shooter": "Tata"})
    _ingest(client, auth, png_bytes, size=64, session=session)
    _ingest(client, auth, png_bytes, size=65)

    assert export(db_session, get_settings().data_path, tmp_path) == 2

    shooters = {
        json.loads((f / "payload.json").read_text())["session"]["shooter"]
        for f in tmp_path.iterdir()
    }

    assert shooters == {"Tata", None}
