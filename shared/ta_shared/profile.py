"""Target profile: ring geometry (server scoring) plus the ArUco marker board and
manual-corner layout (client registration). One versioned JSON artifact per
physical target, so scoring stays reproducible. See internal_docs/implementation.md §4.
"""

import json
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

from pydantic import BaseModel, model_validator


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


class MarkerBoard(BaseModel):
    aruco_dict: str  # e.g. "DICT_4X4_50"
    # marker id -> its 4 corner positions in canonical px (TL, TR, BR, BL)
    markers: dict[int, list[tuple[float, float]]]


class TargetProfile(BaseModel):
    name: str
    version: int
    canon_size_px: int
    n_rings: int
    ring_radii_px: list[int]  # ascending, len == n_rings
    # Diameter of the outer *scoring ring* (ring 1), not of the printed card — it is
    # what scales px to mm. Measure it; ISSF prints the two within 15 mm of each other.
    target_diam_mm: float | None = None
    board: MarkerBoard
    manual_corners_canon: list[tuple[float, float]]  # 4 dst pts for the manual 4-click fallback

    @model_validator(mode="after")
    def _rings_are_scorable(self) -> "TargetProfile":
        check_ring_geometry(self.n_rings, self.ring_radii_px)

        return self


def load_profile(path: str | Path) -> TargetProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    return TargetProfile.model_validate(data)
