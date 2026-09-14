"""The three dashboard views (implementation.md §10).

Everything here reads a session put in by the real ingest path, so a change to how
scoring is persisted fails these too rather than only the ingest tests.
"""

import re

import pytest
from fastapi import Response, status
from sqlmodel import Session, select
from tests.conftest import make_jpeg, make_payload

from target_analyzer.api.endpoints.ingest import _ingest_onto_existing_image
from target_analyzer.config import get_settings
from target_analyzer.models import Image, Interpretation
from target_analyzer.templating import format_metres


def test_the_trend_page_lists_an_ingested_session(auth_client, ingested):
    response = auth_client.get("/dashboard")

    assert response.status_code == status.HTTP_200_OK
    assert "CZ 75" in response.text
    assert f'href="/dashboard/session/{ingested["session_id"]}"' in response.text
    assert "25 m" in response.text


@pytest.mark.parametrize(
    ("metres", "shown"), [(25.0, "25"), (12.5, "12.5"), (7.5, "7.5"), (None, "—")]
)
def test_a_distance_keeps_its_decimals(metres, shown):
    """Formatting to whole metres printed a 12.5 m range as "12 m", and two ranges a
    decimetre apart as one filter option.
    """
    assert format_metres(metres) == shown


@pytest.mark.parametrize("written", ["25", "25.0", "25.00"])
def test_the_distance_filter_stays_selected_however_it_was_written(auth_client, ingested, written):
    """The control has to agree with the table: the filter applies on any spelling, so
    echoing the raw string would leave the select reading "All" over filtered rows.
    """
    response = auth_client.get(f"/dashboard?distance={written}")
    row = f'href="/dashboard/session/{ingested["session_id"]}"'

    assert row in response.text
    assert re.search(r'<option value="25\.0"\s+selected>', response.text)


@pytest.mark.parametrize(
    ("query", "listed"),
    [
        ("", True),
        ("?gun=CZ+75", True),
        ("?gun=Glock", False),
        ("?distance=25", True),
        ("?distance=50", False),
        ("?gun=CZ+75&distance=25", True),
        # A hand-edited URL falls back to no filter rather than 422-ing or 500-ing.
        ("?distance=nonsense", True),
        ("?distance=inf", True),
        ("?distance=%C2%B2", True),
    ],
)
def test_the_trend_filters_by_gun_and_distance(auth_client, ingested, query, listed):
    response = auth_client.get(f"/dashboard{query}")
    # The row's own link, not the gun name: every gun is also an <option> in the filter,
    # so the name being on the page proves nothing about what the table holds.
    row = f'href="/dashboard/session/{ingested["session_id"]}"'

    assert response.status_code == status.HTTP_200_OK
    assert (row in response.text) is listed


def test_the_session_svg_draws_a_ring_per_ring_and_a_marker_per_hole(
    auth_client, ingested, profile
):
    response = auth_client.get(f"/dashboard/session/{ingested['session_id']}")

    assert response.status_code == status.HTTP_200_OK
    assert f'viewBox="0 0 {profile.canon_size_px} {profile.canon_size_px}"' in response.text
    assert response.text.count('class="ring"') == profile.n_rings
    # Both hits of the fixture payload sit inside the innermost ring.
    assert response.text.count('class="hole-good"') == ingested["n_holes"]


def test_the_session_page_prints_the_stored_bias_direction(auth_client, ingested, db_session):
    """Asserting it equals the column, not a literal: a template that re-derived the
    direction would pass against a hard-coded string and still be a second copy of the
    sign convention.
    """
    stored = db_session.get(Interpretation, ingested["interpretation_id"])
    response = auth_client.get(f"/dashboard/session/{ingested['session_id']}")

    assert stored.metrics["bias_direction"] in response.text


@pytest.mark.parametrize(
    ("query", "shown"), [("", True), ("?photo=off", False), ("?photo=on", True)]
)
def test_the_photo_backdrop_is_on_by_default_and_the_link_turns_it_off(
    auth_client, ingested, query, shown
):
    response = auth_client.get(f"/dashboard/session/{ingested['session_id']}{query}")
    backdrop = f'href="/dashboard/image/{ingested["image_id"]}"'

    assert (backdrop in response.text) is shown


def test_saving_notes_lands_back_at_the_notes_box_and_sticks(auth_client, ingested):
    session_id = ingested["session_id"]
    saved = auth_client.post(
        f"/dashboard/session/{session_id}/notes",
        data={"notes": "sore from gym"},
        follow_redirects=False,
    )

    assert saved.status_code == status.HTTP_303_SEE_OTHER
    assert saved.headers["location"] == f"/dashboard/session/{session_id}#notes"
    assert "sore from gym" in auth_client.get(f"/dashboard/session/{session_id}").text
    # The note is the context that stops a bad day reading as a regression, so it has to
    # reach the view the regression would show up in.
    assert "sore from gym" in auth_client.get("/dashboard").text


def test_the_compare_view_lists_every_reading_of_the_image(
    auth_client, ingested, profile, db_session: Session
):
    """Driven below HTTP because ``method`` is a Literal on the wire, so a second
    reading cannot arrive through the endpoint yet.
    """
    ship = make_payload(make_jpeg())
    ship.method = "cv_blob"
    image = db_session.exec(select(Image)).one()
    _ingest_onto_existing_image(db_session, Response(), image, profile, ship)

    response = auth_client.get(f"/dashboard/compare/{ingested['image_id']}")

    assert response.status_code == status.HTTP_200_OK
    assert "manual" in response.text
    assert "cv_blob" in response.text
    # The hand-marked reading is the baseline the others are read against, so it takes
    # the leftmost column rather than wherever its name happens to sort.
    assert response.text.index("manual") < response.text.index("cv_blob")


@pytest.mark.parametrize(
    "path", ["/dashboard/session/999999", "/dashboard/compare/999999", "/dashboard/image/999999"]
)
def test_an_unknown_id_is_a_404(auth_client, ingested, path):
    assert auth_client.get(path).status_code == status.HTTP_404_NOT_FOUND


def test_the_normalized_render_is_served_to_a_logged_in_user(auth_client, ingested, db_session):
    image = db_session.get(Image, ingested["image_id"])
    response = auth_client.get(f"/dashboard/image/{image.id}")

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == "image/png"
    assert response.content == (get_settings().data_path / image.normalized_path).read_bytes()


def test_the_image_route_cannot_be_made_to_serve_a_path_outside_the_data_directory(
    auth_client, ingested, db_session
):
    image = db_session.get(Image, ingested["image_id"])
    image.normalized_path = "../../../../etc/passwd"
    db_session.add(image)
    db_session.commit()

    assert auth_client.get(f"/dashboard/image/{image.id}").status_code == status.HTTP_404_NOT_FOUND


def test_a_render_missing_from_disk_is_a_404_not_a_500(auth_client, ingested, db_session):
    image = db_session.get(Image, ingested["image_id"])
    (get_settings().data_path / image.normalized_path).unlink()

    assert auth_client.get(f"/dashboard/image/{image.id}").status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.parametrize("image_id", ["../../etc/passwd", "1/../../etc/passwd", "abc"])
def test_a_non_integer_image_id_never_reaches_the_handler(auth_client, ingested, image_id):
    response = auth_client.get(f"/dashboard/image/{image_id}")

    assert response.status_code != status.HTTP_200_OK
