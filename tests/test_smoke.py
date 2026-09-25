"""P0 smoke check: the packages are wired and the shared contract round-trips."""

import pytest
from pydantic import ValidationError

import ta_shared
import target_analyzer
from ta_shared.payload import Hit, SessionMeta, ShipPayload
from tests.conftest import make_jpeg, make_payload


def test_packages_import():
    assert ta_shared.__version__
    assert target_analyzer.__version__


def test_ship_payload_roundtrips():
    payload = make_payload(make_jpeg())

    assert ShipPayload.model_validate_json(payload.model_dump_json()) == payload


@pytest.mark.parametrize(
    "path",
    [
        pytest.param((), id="payload"),
        pytest.param(("session",), id="session"),
        pytest.param(("hits", 0), id="hit"),
    ],
)
def test_an_unknown_field_is_refused_at_every_level(path):
    data = make_payload(make_jpeg()).model_dump(mode="json")
    node = data

    for key in path:
        node = node[key]

    node["lane"] = 7

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ShipPayload.model_validate(data)


@pytest.mark.parametrize("coordinate", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_coordinates_never_reach_the_server(coordinate):
    """A non-finite coordinate would score as a miss and then break the JSON that
    carries the metrics — it has to die at the contract.
    """
    with pytest.raises(ValidationError):
        Hit(x_canon=coordinate, y_canon=500.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # Lands in hole.confidence and is rendered; NaN breaks the same JSON the
        # coordinates above are guarded against.
        ("confidence", float("nan")),
        ("confidence", 1.5),
        ("confidence", -0.1),
    ],
)
def test_hit_confidence_is_bounded(field, value):
    with pytest.raises(ValidationError):
        Hit(x_canon=500.0, y_canon=500.0, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # Indexed with `gun` and drives the Trend filter: a non-finite or negative
        # distance matches no filter, ever, so the row silently leaves the dashboard.
        ("distance_m", float("nan")),
        ("distance_m", 0),
        ("distance_m", -25),
        ("target_profile_version", 0),
        ("gun", "x" * 101),
        ("target_profile", "x" * 101),
        ("notes", "x" * 10_001),
    ],
)
def test_session_meta_bounds(field, value):
    fields = {
        "gun": "test",
        "distance_m": 10,
        "target_profile": "issf_precision",
        "target_profile_version": 1,
    }

    with pytest.raises(ValidationError):
        SessionMeta(**(fields | {field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_sha256", "0" * 63),  # truncated
        ("image_sha256", "0" * 65),  # too long
        ("image_sha256", "Z" * 64),  # not hex
        ("image_sha256", "A" * 64),  # uppercase: a second spelling of one identity
        ("canon_size_px", 0),
        # 200 holes is already far past a real string; unbounded means one payload
        # becomes a million hole INSERTs in a single transaction.
        ("hits", [Hit(x_canon=500, y_canon=500)] * 201),
    ],
)
def test_ship_payload_bounds(field, value):
    fields = {
        "image_sha256": "0" * 64,
        "canon_size_px": 1000,
        "session": SessionMeta(
            gun="test",
            distance_m=10,
            target_profile="issf_precision",
            target_profile_version=1,
        ),
        "hits": [Hit(x_canon=500, y_canon=500)],
    }

    with pytest.raises(ValidationError):
        ShipPayload(**(fields | {field: value}))


SESSION = {"gun": "g", "distance_m": 10, "target_profile": "p", "target_profile_version": 1}


@pytest.mark.parametrize(
    ("typed", "stored"),
    [(None, None), ("Tata", "Tata"), ("  Tata ", "Tata"), ("   ", None), ("Zażółć", "Zażółć")],
)
def test_a_shooter_is_stored_as_typed_and_blank_is_the_owner(typed, stored):
    assert SessionMeta(**SESSION, shooter=typed).shooter == stored


@pytest.mark.parametrize("typed", ["-", " - ", "x" * 101])
def test_a_shooter_needs_a_letter_or_digit_and_a_bounded_length(typed):
    with pytest.raises(ValidationError):
        SessionMeta(**SESSION, shooter=typed)
