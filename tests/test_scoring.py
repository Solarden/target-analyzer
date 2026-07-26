"""Scoring and metrics (implementation.md §6).

The fixture profile is the one from the doc — canon 1000, ten rings every 50 px,
a 500 mm outer ring — chosen so ``mm_per_px == 0.5`` and every expected number
below is checkable by hand.

A ``SimpleNamespace`` stands in for a profile on purpose: scoring is typed against
the :class:`~target_analyzer.scoring.RingGeometry` protocol, so the same functions
take the JSON profile *and* the DB row. Anything with these four attributes works.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ta_shared.profile import load_profile
from target_analyzer.models import Interpretation, TargetProfile
from target_analyzer.scoring import (
    ScoreResult,
    bias_direction,
    compute_metrics,
    mm_per_px,
    ring_for_hit,
    selfcheck,
)

PROFILE = SimpleNamespace(
    canon_size_px=1000,
    n_rings=10,
    ring_radii_px=[50 * i for i in range(1, 11)],
    target_diam_mm=500.0,
)
UNCALIBRATED = SimpleNamespace(
    canon_size_px=1000,
    n_rings=10,
    ring_radii_px=[50 * i for i in range(1, 11)],
    target_diam_mm=None,
)

# The doc's worked example: a tight group down-and-right of centre.
GROUP = [(500, 500), (510, 500), (500, 520)]

SEEDED_PROFILE = Path(__file__).parent.parent / "profiles" / "issf_precision.json"


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (500, 500, 10),  # dead centre
        (500, 549, 10),  # just inside the innermost line
        (500, 550, 10),  # exactly on the 50 px line -> the higher ring
        (500, 600, 9),  # exactly on the 100 px line -> 9, not 8
        (500, 625, 8),  # mid-annulus, 100..150
        (800, 900, 1),  # 3-4-5 off centre: d == 500, both axes count
        (500, 1000, 1),  # on the outermost line -> still a score
        (500, 1001, 0),  # a pixel past it -> miss
        (0, 0, 0),  # nowhere near -> miss
    ],
)
def test_ring_for_hit(x, y, expected):
    assert ring_for_hit(x, y, PROFILE) == expected


def test_group_metrics_match_the_hand_computed_fixture():
    result = compute_metrics(GROUP, PROFILE)

    assert result.rings == (10, 10, 10)
    assert (result.n_holes, result.total_score, result.best_score) == (3, 30, 10)
    assert result.avg_score == 10.0
    assert result.centroid == pytest.approx((503.33, 506.67), abs=0.01)
    assert result.extreme_spread == pytest.approx(22.36, abs=0.01)  # sqrt(500)
    assert result.extreme_spread_mm == pytest.approx(11.18, abs=0.01)
    assert result.mean_radius == pytest.approx(10.21, abs=0.01)
    assert result.sigma_x == pytest.approx(4.71, abs=0.01)
    assert result.sigma_y == pytest.approx(9.43, abs=0.01)


def test_bias_of_a_low_right_group_reads_low_right():
    """The §1 sign guard: x right, y down. A flipped axis inverts the coaching."""
    result = compute_metrics(GROUP, PROFILE)

    assert result.bias_vec == pytest.approx((3.33, 6.67), abs=0.01)
    assert result.bias_vec[0] > 0 and result.bias_vec[1] > 0
    assert result.bias == pytest.approx(7.45, abs=0.01)
    assert result.bias_mm == pytest.approx(3.73, abs=0.01)
    assert bias_direction(result.bias_vec) == "low-right"


@pytest.mark.parametrize(
    ("bias_vec", "expected"),
    [
        ((3.33, 6.67), "low-right"),
        ((-4.0, -4.0), "high-left"),
        ((0.0, 8.0), "low"),
        ((8.0, 0.0), "right"),
        ((0.0, -8.0), "high"),
        ((0.1, -0.2), "centered"),  # sub-pixel is not a direction
        (None, None),
    ],
)
def test_bias_direction(bias_vec, expected):
    assert bias_direction(bias_vec) == expected


def test_no_hits_is_not_a_zero_sized_group():
    result = compute_metrics([], PROFILE)

    assert (result.n_holes, result.total_score, result.rings) == (0, 0, ())
    assert result.avg_score is None
    assert result.best_score is None
    assert result.centroid is None
    assert result.extreme_spread is None
    assert result.mean_radius is None
    assert result.bias_mm is None


def test_single_hit_has_a_bias_but_no_spread():
    result = compute_metrics([(520, 480)], PROFILE)

    assert result.extreme_spread == 0.0
    assert result.mean_radius == 0.0
    assert (result.sigma_x, result.sigma_y) == (0.0, 0.0)
    assert bias_direction(result.bias_vec) == "high-right"


def test_uncalibrated_profile_reports_px_only():
    result = compute_metrics(GROUP, UNCALIBRATED)

    assert mm_per_px(UNCALIBRATED) is None
    assert result.extreme_spread == pytest.approx(22.36, abs=0.01)  # px is always there
    assert result.extreme_spread_mm is None
    assert result.bias_mm is None
    assert result.headline_columns()["mean_radius_mm"] is None


def test_headline_columns_are_real_interpretation_columns():
    """Guards the seam: ingest splats these straight into the row (§7)."""
    columns = compute_metrics(GROUP, PROFILE).headline_columns()

    assert set(columns) <= set(Interpretation.model_fields)


def test_metrics_jsonb_survives_a_json_round_trip():
    metrics = compute_metrics(GROUP, PROFILE).metrics_jsonb()

    assert json.loads(json.dumps(metrics)) == metrics


def test_selfcheck_passes():
    """The framework-free `python -m target_analyzer.scoring` path, under pytest too."""
    selfcheck()


def test_the_seeded_profile_is_scorable():
    """The shipped artifact must load (validator included) and score sanely."""
    profile = load_profile(SEEDED_PROFILE)
    centre = profile.canon_size_px / 2
    result = compute_metrics([(centre, centre)], profile)

    assert isinstance(result, ScoreResult)
    assert result.rings == (profile.n_rings,)
    assert bias_direction(result.bias_vec) == "centered"
    assert result.mm_per_px == pytest.approx(profile.target_diam_mm / (2 * 500))


def test_a_database_row_scores_identically_to_the_json_profile():
    """The RingGeometry seam: ingest scores the DB row directly, never a conversion
    of it back into the JSON shape. Needs no database — only the attributes.
    """
    profile = load_profile(SEEDED_PROFILE)
    row = TargetProfile(
        name=profile.name,
        version=profile.version,
        n_rings=profile.n_rings,
        canon_size_px=profile.canon_size_px,
        ring_radii_px=profile.ring_radii_px,
        target_diam_mm=profile.target_diam_mm,
        board=profile.board.model_dump(),
    )

    assert compute_metrics(GROUP, row) == compute_metrics(GROUP, profile)


@pytest.mark.parametrize(
    "ring_radii_px",
    [
        [50 * i for i in range(1, 10)],  # one radius short of n_rings
        [50, 100, 150, 250, 200, 300, 350, 400, 450, 500],  # 250 and 200 swapped
        [],
    ],
)
def test_unscorable_geometry_raises_instead_of_mis_scoring(ring_radii_px):
    """A hand-inserted profile row is not validated by pydantic — scoring has to
    refuse it rather than quietly score every hit against the wrong ring (§1).
    """
    broken = SimpleNamespace(
        canon_size_px=1000, n_rings=10, ring_radii_px=ring_radii_px, target_diam_mm=500.0
    )

    with pytest.raises(ValueError):
        compute_metrics(GROUP, broken)
