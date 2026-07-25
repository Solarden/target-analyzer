"""Target profile: ring geometry (server scoring) plus the ArUco marker board and
manual-corner layout (client registration). One versioned JSON artifact per
physical target, so scoring stays reproducible. See internal_docs/implementation.md §4.
"""

import json
from pathlib import Path

from pydantic import BaseModel


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
    target_diam_mm: float | None = None  # outer scoring-ring diameter -> mm metrics
    board: MarkerBoard  # client registration (ArUco)
    manual_corners_canon: list[tuple[float, float]]  # 4 dst pts for the manual 4-click fallback


def load_profile(path: str | Path) -> TargetProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    return TargetProfile.model_validate(data)
