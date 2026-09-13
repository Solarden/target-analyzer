"""Marking the holes. See internal_docs/implementation.md §9 and §13.

Clicks land on the normalized image, so they already *are* canonical pixels — nothing
here converts coordinates. This is also the seam Phase 2 replaces: a detector is a
function from the normalized image to a list of hits, and ``manual`` is the first one.
"""

import numpy as np

from ta_client import pick
from ta_client.board import draw_rings
from ta_shared.payload import MAX_HITS, Hit
from ta_shared.profile import TargetProfile


def pick_hits(normalized: np.ndarray, profile: TargetProfile) -> list[Hit] | None:
    """Click every hole; ``None`` if cancelled. Confidence is ``None`` for a human."""
    # Stop at the contract's cap here rather than letting the server reject a payload
    # after a whole string has been clicked.
    points = pick.pick_points(
        draw_rings(normalized, profile), "Click each hole", max_points=MAX_HITS
    )

    if points is None:
        return None

    return [Hit(x_canon=x, y_canon=y) for x, y in points]
