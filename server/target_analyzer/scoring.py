"""Scoring and metrics — pure functions of (hits, profile). No DB, no FastAPI.

See internal_docs/implementation.md §1 (the canonical-frame contract) and §6.

The client never scores: it warps a photo into the profile's canonical square and
sends hit coordinates in that frame. Everything here turns those coordinates into
rings and metrics, so re-measuring a target (a new profile version) re-scores
every past session consistently.

Two invariants this module owns, both guarded by :func:`selfcheck`:

- **A hit on a ring line scores up.** ``bisect_left`` over the ascending radii
  puts a boundary hit in the *higher* ring, the way a scoring gauge does.
- **Canonical pixels are image convention** — x grows right, y grows **down**. A
  group that is physically low-and-right must read low-and-right. The translation
  from that to a shooter's words lives in :func:`bias_direction` and **nowhere
  else**; a second copy of the sign convention is how a bias read gets inverted.

Run ``python -m target_analyzer.scoring`` to check both without pytest.
"""

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import pstdev
from types import SimpleNamespace
from typing import Protocol

from ta_shared.profile import check_ring_geometry


class RingGeometry(Protocol):
    """What scoring needs from "a profile" — the ring geometry, nothing else.

    Satisfied structurally by both :class:`ta_shared.profile.TargetProfile` (the
    JSON artifact, client-side and at rest) and
    :class:`target_analyzer.models.TargetProfile` (the DB row the ingest endpoint
    loads). Neither ever needs converting into the other to be scored.
    """

    canon_size_px: int
    n_rings: int
    ring_radii_px: list[int]
    target_diam_mm: float | None


def center(profile: RingGeometry) -> tuple[float, float]:
    """The Point of Aim: the target centre is the centre of the canonical frame."""
    half = profile.canon_size_px / 2

    return half, half


def mm_per_px(profile: RingGeometry) -> float | None:
    """Millimetres per canonical pixel, or None when the profile has no physical
    diameter — then only px metrics are reported.

    ``target_diam_mm`` spans the **outer scoring ring**, i.e. twice its radius.
    """
    if profile.target_diam_mm is None:
        return None

    return profile.target_diam_mm / (2 * profile.ring_radii_px[-1])


def ring_for_hit(x: float, y: float, profile: RingGeometry) -> int:
    """Score one hit: ``n_rings`` at the centre down to 1 at the outer ring, 0 = miss.

    A hit exactly on a ring line scores up (the higher ring), which is what
    ``bisect_left`` on ascending radii gives for free.
    """
    cx, cy = center(profile)
    d = math.hypot(x - cx, y - cy)
    i = bisect_left(profile.ring_radii_px, d)

    # >= rather than == so a malformed profile (more radii than rings) reads as a
    # miss instead of a negative ring.
    return 0 if i >= profile.n_rings else profile.n_rings - i


def bias_direction(bias_vec: tuple[float, float] | None, tol: float = 0.5) -> str | None:
    """The bias vector in a shooter's words: "low-right", "high", "centered", ….

    The single place the canonical sign convention (§1) is translated: +x is
    right, +y is **down** hence *low*. Everything else — dashboard, coaching
    text — asks this function rather than re-deciding.

    ``tol`` is half a canonical pixel: hits are placed by hand, so a smaller
    offset than that is not a real direction.
    """
    if bias_vec is None:
        return None

    dx, dy = bias_vec
    vertical = "low" if dy > tol else "high" if dy < -tol else ""
    horizontal = "right" if dx > tol else "left" if dx < -tol else ""

    return "-".join(part for part in (vertical, horizontal) if part) or "centered"


@dataclass(frozen=True)
class ScoreResult:
    """Everything derivable from one set of hits plus one profile.

    Lengths are canonical pixels; the ``_mm`` properties convert with
    ``mm_per_px`` and are None when the profile carries no physical diameter.
    With no hits every measurement is None (``total_score`` is a true 0) — an
    empty string of shots is not a zero-sized group.
    """

    rings: tuple[int, ...]
    n_holes: int
    total_score: int
    avg_score: float | None
    best_score: int | None
    centroid: tuple[float, float] | None  # the Point of Impact
    bias_vec: tuple[float, float] | None  # centroid - centre: accuracy
    bias: float | None  # magnitude of the above
    extreme_spread: float | None  # the classic group size (outlier-driven)
    mean_radius: float | None  # mean distance from the centroid: stable precision
    sigma_x: float | None  # windage dispersion: trigger/grip
    sigma_y: float | None  # elevation dispersion: breathing/anticipation
    mm_per_px: float | None

    def _mm(self, px: float | None) -> float | None:
        if px is None or self.mm_per_px is None:
            return None

        return px * self.mm_per_px

    @property
    def bias_mm(self) -> float | None:
        return self._mm(self.bias)

    @property
    def extreme_spread_mm(self) -> float | None:
        return self._mm(self.extreme_spread)

    @property
    def mean_radius_mm(self) -> float | None:
        return self._mm(self.mean_radius)

    @property
    def sigma_x_mm(self) -> float | None:
        return self._mm(self.sigma_x)

    @property
    def sigma_y_mm(self) -> float | None:
        return self._mm(self.sigma_y)

    @property
    def bias_vec_mm(self) -> tuple[float, float] | None:
        if self.bias_vec is None or self.mm_per_px is None:
            return None
        dx, dy = self.bias_vec

        return dx * self.mm_per_px, dy * self.mm_per_px

    def headline_columns(self) -> dict:
        """The cached columns on ``interpretation`` — what Trend reads directly."""
        return {
            "n_holes": self.n_holes,
            "total_score": self.total_score,
            "avg_score": self.avg_score,
            "best_score": self.best_score,
            "extreme_spread_mm": self.extreme_spread_mm,
            "mean_radius_mm": self.mean_radius_mm,
            "bias_mm": self.bias_mm,
        }

    def metrics_jsonb(self) -> dict:
        """The full metric set for ``interpretation.metrics`` — JSON-safe.

        Keeps the px variants (always present) next to the mm ones (present only
        with a calibrated profile), so a profile without a physical diameter
        still records everything measurable.
        """
        return {
            "rings": list(self.rings),
            "centroid": list(self.centroid) if self.centroid is not None else None,
            "bias_vec": list(self.bias_vec) if self.bias_vec is not None else None,
            "bias_direction": bias_direction(self.bias_vec),
            "mm_per_px": self.mm_per_px,
            "px": {
                "bias": self.bias,
                "extreme_spread": self.extreme_spread,
                "mean_radius": self.mean_radius,
                "sigma_x": self.sigma_x,
                "sigma_y": self.sigma_y,
            },
            "mm": {
                "bias": self.bias_mm,
                "bias_vec": list(self.bias_vec_mm) if self.bias_vec_mm is not None else None,
                "extreme_spread": self.extreme_spread_mm,
                "mean_radius": self.mean_radius_mm,
                "sigma_x": self.sigma_x_mm,
                "sigma_y": self.sigma_y_mm,
            },
        }


def compute_metrics(hits: Sequence[tuple[float, float]], profile: RingGeometry) -> ScoreResult:
    """Score every hit and derive the precision (how tight) and accuracy (where)
    numbers. Pure — same inputs, same output, no I/O.

    Raises ValueError on a profile whose geometry cannot be scored. This is the one
    chokepoint every interpretation passes through, and the DB row it usually scores
    against is hand-inserted at deploy time (§11) — so the check has to be here and
    not only on the JSON model.
    """
    check_ring_geometry(profile.n_rings, profile.ring_radii_px)

    rings = tuple(ring_for_hit(x, y, profile) for x, y in hits)
    scale = mm_per_px(profile)
    n = len(hits)

    if not n:
        return ScoreResult(
            rings=(),
            n_holes=0,
            total_score=0,
            avg_score=None,
            best_score=None,
            centroid=None,
            bias_vec=None,
            bias=None,
            extreme_spread=None,
            mean_radius=None,
            sigma_x=None,
            sigma_y=None,
            mm_per_px=scale,
        )

    xs = [x for x, _ in hits]
    ys = [y for _, y in hits]
    centroid = (sum(xs) / n, sum(ys) / n)
    cx, cy = center(profile)
    bias_vec = (centroid[0] - cx, centroid[1] - cy)

    # default=0.0: a single hit is a group of size zero, not an error.
    extreme_spread = max((math.dist(a, b) for a, b in combinations(hits, 2)), default=0.0)
    # From the centroid, not the target centre — this is precision, not accuracy.
    mean_radius = sum(math.dist(h, centroid) for h in hits) / n

    return ScoreResult(
        rings=rings,
        n_holes=n,
        total_score=sum(rings),
        avg_score=sum(rings) / n,
        best_score=max(rings),
        centroid=centroid,
        bias_vec=bias_vec,
        bias=math.hypot(*bias_vec),
        extreme_spread=extreme_spread,
        mean_radius=mean_radius,
        sigma_x=pstdev(xs),
        sigma_y=pstdev(ys),
        mm_per_px=scale,
    )


def selfcheck() -> None:
    """Assert the frame contract holds. Framework-free on purpose, so it runs on a
    box with no pytest: ``python -m target_analyzer.scoring``.

    The full table of cases lives in tests/test_scoring.py, which calls this too.
    """
    profile = SimpleNamespace(
        canon_size_px=1000,
        n_rings=10,
        ring_radii_px=[50 * i for i in range(1, 11)],
        target_diam_mm=500.0,  # -> mm_per_px == 0.5, so the numbers below are checkable by hand
    )
    assert mm_per_px(profile) == 0.5

    assert ring_for_hit(500, 500, profile) == 10  # dead centre
    assert ring_for_hit(500, 600, profile) == 9  # exactly on the 100 px line -> scores UP
    assert ring_for_hit(500, 1000, profile) == 1  # on the outermost line -> still a score
    assert ring_for_hit(0, 0, profile) == 0  # outside every ring -> miss

    # These three hits sit right of and below the centre, so the bias must read
    # low-right — a flipped sign anywhere in the chain fails here (§1).
    result = compute_metrics([(500, 500), (510, 500), (500, 520)], profile)
    assert result.bias_vec[0] > 0, "positive x must stay 'right'"
    assert result.bias_vec[1] > 0, "positive y must stay 'low' (image convention)"
    assert bias_direction(result.bias_vec) == "low-right"
    assert round(result.extreme_spread, 2) == 22.36
    assert round(result.extreme_spread_mm, 2) == 11.18

    empty = compute_metrics([], profile)
    assert (empty.n_holes, empty.total_score) == (0, 0)
    assert (empty.avg_score, empty.best_score, empty.mean_radius) == (None, None, None)

    print("scoring self-check OK")


if __name__ == "__main__":
    selfcheck()
