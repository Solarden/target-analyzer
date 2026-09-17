"""Marking the holes. See implementation.md §9 and §13.

Clicks land on the normalized image, so they already *are* canonical pixels — nothing
here converts coordinates. This is also the detector seam: a detector is a function from
the normalized image to a list of hits, and its output arrives here as a proposal for a
human to correct.
"""

import numpy as np

from ta_client import pick
from ta_client.board import draw_rings
from ta_shared.payload import MAX_HITS, Hit
from ta_shared.profile import TargetProfile


def pick_hits(
    normalized: np.ndarray, profile: TargetProfile, proposal: list[Hit] | None = None
) -> list[Hit] | None:
    """Confirm every hole, starting from ``proposal``; ``None`` if cancelled.

    Confidence is ``None`` on everything that comes back: a person has looked at each
    point, so the detector's number does not describe it.
    """
    title = "Click each hole" if not proposal else f"{len(proposal)} proposed — fix and confirm"
    # Stop at the contract's cap here rather than letting the server reject a payload
    # after a whole string has been clicked.
    points = pick.pick_points(
        draw_rings(normalized, profile),
        title,
        max_points=MAX_HITS,
        initial=[(hit.x_canon, hit.y_canon) for hit in proposal or []],
    )

    if points is None:
        return None

    return [Hit(x_canon=x, y_canon=y) for x, y in points]
