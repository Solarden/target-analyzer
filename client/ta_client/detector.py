"""What a detector is, and how one is measured. See implementation.md §9 and §13.

A detector is a module, not a function, because a reading is three things the runner needs:

``detect(normalized, profile) -> list[Hit]``
    The reading itself, in canonical pixels. Raises :class:`DetectorError` when it cannot
    produce one, and ``ValueError`` when it is handed something it should never be handed
    — an image that is not the profile's canonical square is a bug upstream, not a bad day.

``check(settings, profile) -> None``
    Everything ``detect`` needs that can be known before the photo is touched. The runner
    calls it before the first window opens, because a detector that refuses after forty
    clicks has thrown those clicks away.

``model_name(settings) -> str | None``
    Which model produced the reading, for the interpretation's ``model`` column. ``None``
    where the detector is the algorithm.
"""

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from ta_shared.agreement import MATCH_TOL_MM, agreement
from ta_shared.payload import Hit
from ta_shared.profile import TargetProfile, load_profile, mm_per_px

PROFILES = Path(__file__).resolve().parents[2] / "profiles"

Detect = Callable[[np.ndarray, TargetProfile], list[Hit]]


class DetectorError(RuntimeError):
    """This detector cannot produce a reading now.

    Distinct from ``ValueError`` on purpose. A missing endpoint, a sleeping box or an
    answer that is not a reading are all conditions the run should survive — the person
    still marks the holes by hand and that reading still ships. A ``ValueError`` from a
    detector means it was called wrongly, and should stop the run.
    """


def load_session(session: Path, profile: Path | None) -> tuple[dict, TargetProfile]:
    """A stored session's payload, and the profile it was scored against."""
    marked = session / "payload.json"

    if not marked.is_file():
        raise SystemExit(f"no payload.json in {session}")

    payload = json.loads(marked.read_text(encoding="utf-8"))
    name = payload["session"]["target_profile"]

    # A session folder can be written by another machine, and the payload names its own
    # profile — so the name selects a file in PROFILES rather than reaching anywhere.
    if Path(name).name != name:
        raise SystemExit(f"{name!r} is not a profile name")

    path = profile or PROFILES / f"{name}.json"

    if not path.is_file():
        raise SystemExit(f"no profile at {path} — pass --profile")

    return payload, load_profile(path)


def report(session: Path, payload: dict, profile: TargetProfile, detect: Detect) -> None:
    """Measure one stored session: what a detector finds against what a person marked."""
    normalized = cv2.imread(str(session / "normalized.png"))

    if normalized is None:
        raise SystemExit(f"no readable normalized.png in {session}")

    truth = [(hit["x_canon"], hit["y_canon"]) for hit in payload["hits"]]

    try:
        proposal = detect(normalized, profile)
    except (DetectorError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    result = agreement(
        truth,
        [(hit.x_canon, hit.y_canon) for hit in proposal],
        MATCH_TOL_MM / mm_per_px(profile),
    )
    offset = "—" if result.mean_offset_px is None else f"{result.mean_offset_px:.1f} px"

    print(f"{session.name}: {len(truth)} marked by hand, {len(proposal)} proposed")
    print(f"  matched {result.matched} · missed {result.missed} · spurious {result.spurious}")
    print(f"  mean offset over the matched {offset}")
