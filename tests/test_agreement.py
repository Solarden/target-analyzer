"""Matching one reading of a photo against another."""

import pytest

from ta_shared.agreement import agreement

TRUTH = [(100.0, 100.0), (200.0, 200.0), (300.0, 300.0)]


@pytest.mark.parametrize(
    ("other", "matched", "missed", "spurious"),
    [
        ([], 0, 3, 0),
        (TRUTH, 3, 0, 0),
        ([(103.0, 100.0)], 1, 2, 0),
        ([(100.0, 100.0), (900.0, 900.0)], 1, 2, 1),
        # Just outside the tolerance is a different hole, not a badly placed one.
        ([(111.0, 100.0)], 0, 3, 1),
        # Two detections of one hole: the closer takes it and the other is spurious.
        ([(101.0, 100.0), (104.0, 100.0)], 1, 2, 1),
    ],
)
def test_what_matches_and_what_does_not(other, matched, missed, spurious):
    result = agreement(TRUTH, other, tol_px=10.0)

    assert (result.matched, result.missed, result.spurious) == (matched, missed, spurious)


def test_each_hole_takes_its_own_detection():
    result = agreement([(100.0, 100.0), (106.0, 100.0)], [(106.5, 100.0), (100.5, 100.0)], 10.0)

    assert (result.matched, result.mean_offset_px) == (2, 0.5)


def test_nothing_to_measure_leaves_no_offset():
    result = agreement(TRUTH, [(900.0, 900.0)], tol_px=10.0)

    assert result.mean_offset_px is None
