"""Geometry the target overlay is drawn with, shared by Session and Compare.

Everything here is in canonical pixels, which is also the SVG user space: the
``viewBox`` is the canonical square, so x grows right and y grows **down** on both
sides and nothing is ever flipped (§1). A template that "corrects" y is the bug.
"""

from dataclasses import dataclass

from ta_shared.profile import RingGeometry, mm_per_px
from target_analyzer.scoring import center

# Drawn marks only have to be close enough to sit on the real hole in the photo, and
# this is the calibre the target is shot with.
NOMINAL_HOLE_DIAM_MM = 9.0


@dataclass(frozen=True)
class Geometry:
    canon: int
    centre_x: float  # the Point of Aim, and the origin of the bias line
    centre_y: float
    hole_radius: float
    stroke: float


def geometry(profile: RingGeometry) -> Geometry:
    """Drawing sizes for one profile, scaled so a 1000 px and a 1500 px frame look alike."""
    scale = mm_per_px(profile)
    # Without a physical scale there is no true hole size — fall back to something
    # visible rather than drawing nothing.
    hole_radius = NOMINAL_HOLE_DIAM_MM / 2 / scale if scale else profile.canon_size_px / 150
    # Asked, not recomputed: a second definition of the frame's centre is what puts
    # the rings and the bias line where the scorer does not agree they are.
    centre_x, centre_y = center(profile)

    return Geometry(
        canon=profile.canon_size_px,
        centre_x=centre_x,
        centre_y=centre_y,
        hole_radius=hole_radius,
        stroke=profile.canon_size_px / 500,
    )
