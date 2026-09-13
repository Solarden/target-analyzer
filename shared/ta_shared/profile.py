"""Target profile: ring geometry (server scoring) plus the ArUco marker board and
manual-corner layout (client registration). One versioned JSON artifact per
physical target, so scoring stays reproducible. See internal_docs/implementation.md §4.
"""

import json
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, model_validator


def check_ring_geometry(n_rings: int, ring_radii_px: Sequence[int]) -> None:
    """Raise ValueError unless the radii can be scored at all.

    Scoring reads them positionally (ring = n_rings - bisect index), so a wrong
    count or a non-ascending list does not fail — it silently mis-scores every
    hit. Lives here, not in the model, because the *other* thing that gets scored
    is the database row (hand-inserted at deploy time), and one rule beats two
    copies of it.
    """
    if len(ring_radii_px) != n_rings:
        raise ValueError(f"ring_radii_px has {len(ring_radii_px)} entries, expected {n_rings}")

    if any(inner >= outer for inner, outer in pairwise(ring_radii_px)):
        raise ValueError("ring_radii_px must be strictly ascending (centre outward)")


class RingGeometry(Protocol):
    """What scoring needs from "a profile" — the ring geometry, nothing else.

    Satisfied structurally by both :class:`TargetProfile` (the JSON artifact, client-side
    and at rest) and :class:`target_analyzer.models.TargetProfile` (the DB row the ingest
    endpoint loads). Neither ever needs converting into the other to be scored.
    """

    canon_size_px: int
    n_rings: int
    ring_radii_px: list[int]
    target_diam_mm: float | None


def mm_per_px(profile: RingGeometry) -> float | None:
    """Millimetres per canonical pixel, or None when the profile has no physical
    diameter — then only px metrics are reported.

    ``target_diam_mm`` spans the **outer scoring ring**, i.e. twice its radius. Shared
    because both sides need this scale: the server to report metrics in millimetres, the
    client to print a sheet at its true size and to judge a registration in them.
    """
    if profile.target_diam_mm is None:
        return None

    return profile.target_diam_mm / (2 * profile.ring_radii_px[-1])


class MarkerBoard(BaseModel):
    aruco_dict: str  # e.g. "DICT_4X4_50"
    # marker id -> its 4 corner positions in canonical px (TL, TR, BR, BL)
    markers: dict[int, list[tuple[float, float]]]


class TargetProfile(BaseModel):
    # Constrained because it is not only a label: the client builds the printable sheet's
    # filename out of it and writes the name into the sheet itself.
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    version: int
    canon_size_px: int
    n_rings: int
    ring_radii_px: list[int]  # ascending, len == n_rings
    # Diameter of the outer *scoring ring* (ring 1), not of the printed card — it is
    # what scales px to mm. Measure it; ISSF prints the two within 15 mm of each other.
    target_diam_mm: float | None = None
    # The ring whose outer edge bounds the black aiming area, where a target has one: the
    # largest circle of known radius on the paper, and so the client's frame reference.
    black_ring: int | None = None
    board: MarkerBoard
    # Exactly four, matching the four corners the client's fallback asks a human to
    # click; a shorter list would only fail deep inside the homography fit.
    manual_corners_canon: list[tuple[float, float]] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def _rings_are_scorable(self) -> "TargetProfile":
        check_ring_geometry(self.n_rings, self.ring_radii_px)

        if self.black_ring is not None and not 1 <= self.black_ring <= self.n_rings:
            raise ValueError(f"black_ring {self.black_ring} is not one of the {self.n_rings} rings")

        return self

    def ring_radius_px(self, ring: int) -> int:
        """The outer radius of a scoring ring, counting the way a shooter does.

        ``ring_radii_px`` runs centre outward, so ring 10 is first and ring 1 last.
        """
        return self.ring_radii_px[self.n_rings - ring]


def load_profile(path: str | Path) -> TargetProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    return TargetProfile.model_validate(data)
