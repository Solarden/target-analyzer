"""P0 smoke check: the packages are wired and the shared contract round-trips."""

from datetime import date

import pytest
from pydantic import ValidationError

import ta_shared
import target_analyzer
from ta_shared.payload import Hit, SessionMeta, ShipPayload


def test_packages_import():
    assert ta_shared.__version__
    assert target_analyzer.__version__


def test_ship_payload_roundtrips():
    payload = ShipPayload(
        image_sha256="0" * 64,
        canon_size_px=1000,
        session=SessionMeta(
            gun="test",
            distance_m=10,
            shot_at=date(2026, 7, 24),
            target_profile="issf_precision",
            target_profile_version=1,
        ),
        hits=[Hit(x_canon=500, y_canon=500)],
    )

    assert ShipPayload.model_validate_json(payload.model_dump_json()) == payload


@pytest.mark.parametrize("coordinate", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_coordinates_never_reach_the_server(coordinate):
    """A non-finite coordinate would score as a miss and then break the JSON that
    carries the metrics — it has to die at the contract.
    """
    with pytest.raises(ValidationError):
        Hit(x_canon=coordinate, y_canon=500.0)
