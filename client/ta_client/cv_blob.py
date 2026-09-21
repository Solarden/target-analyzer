"""Classic-CV hole detection. See implementation.md §9 and §13.

A bullet hole is a blob of known size — the calibre, in canonical pixels — so a
Laplacian of Gaussian tuned to that size responds to holes and largely ignores the
printed rings, which are thin lines at a different scale.

The response is taken as a **magnitude**, not a signed dark-blob score. On paper a hole
is darker than its surroundings; inside the black aiming area the torn edge catches the
light and it is brighter. Taking either sign finds both.

What this cannot do is tell a hole from *last* session's hole. On a patched target the
tape, the old holes showing through and the printed digits are all blobs of the same
size, and no local rule separates them — so this proposes, and a person decides
(`hits.pick_hits`). Two holes closer together than one calibre also arrive as one
proposal, because the suppression below keeps a single peak per hole's width.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from ta_client.config import Settings
from ta_client.detector import DetectorError
from ta_shared.agreement import MATCH_TOL_MM, agreement
from ta_shared.payload import MAX_HITS, Hit
from ta_shared.profile import TargetProfile, load_profile, mm_per_px

# 5.6 mm is the smallest calibre this is used with, and a LoG tuned small still answers a
# larger hole — the other way round it splits one hole into an annulus of weak peaks.
DEFAULT_HOLE_DIAM_MM = 5.6
# Of the strongest response in the frame, so the cut travels with the exposure rather
# than with an absolute grey level. Leaning high: a proposal a person has to delete costs
# more of their attention than one they have to add.
MIN_RESPONSE_FRACTION = 0.7
# The ratio between the two blurs. Wide enough that the surround is really surround, tight
# enough that a neighbouring hole is not part of it.
SURROUND_RATIO = 2.5


def model_name(_settings: Settings | None = None) -> None:
    """No model produced this reading: the detector is the algorithm."""
    return None


def check(_settings: Settings | None, profile: TargetProfile) -> None:
    """Raise what ``detect`` would raise, early enough that no work is lost to it.

    Takes settings it does not read, so the runner can ask every detector the same
    question. ``None`` is the honest argument from a caller that has none to give.
    """
    if mm_per_px(profile) is None:
        raise DetectorError(
            f"{profile.name} v{profile.version} has no target_diam_mm, so a hole has no "
            "size in canonical pixels"
        )


def detect(
    normalized: np.ndarray,
    profile: TargetProfile,
    hole_diam_mm: float = DEFAULT_HOLE_DIAM_MM,
) -> list[Hit]:
    """Propose holes in a canonical-frame image, strongest first.

    Raises ``ValueError`` when the profile carries no physical scale, or the image is not
    its canonical square. Both are things this needs and cannot infer, and a guess at
    either mis-tunes every step below in silence.
    """
    check(None, profile)
    scale = mm_per_px(profile)
    canon = profile.canon_size_px

    # Every measurement below is in the profile's geometry, so an image that is not its
    # canonical square would be masked and scaled against rings that are somewhere else.
    # Channels are checked here too, so one exit covers everything this cannot read.
    if normalized.ndim != 3 or normalized.shape[:2] != (canon, canon):
        shape = "x".join(str(n) for n in normalized.shape)

        raise ValueError(
            f"image is {shape}, not {profile.name} v{profile.version}'s "
            f"{canon}x{canon}x3 canonical square"
        )

    diameter = hole_diam_mm / scale
    sigma = diameter / 2 / np.sqrt(2)
    grey = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    response = np.abs(
        cv2.GaussianBlur(grey, (0, 0), sigma)
        - cv2.GaussianBlur(grey, (0, 0), sigma * SURROUND_RATIO)
    )
    # Outside the outermost scoring ring is off the target: paper edge, backing board and
    # whatever the photo caught behind it, all of which answer a blob filter.
    on_target = np.zeros((canon, canon), np.uint8)
    cv2.circle(on_target, (canon // 2, canon // 2), profile.ring_radii_px[-1], 255, -1)
    response[on_target == 0] = 0.0

    # One peak per hole: a dilation by the hole's own size keeps only the local maximum,
    # so a single hole cannot arrive as a cluster of neighbouring pixels.
    window = int(diameter) | 1
    peaks = response == cv2.dilate(response, np.ones((window, window), np.uint8))
    strongest = float(response.max())

    if strongest <= 0:
        return []

    found = [
        (float(response[y, x]), float(x), float(y))
        for y, x in np.argwhere(peaks & (response >= MIN_RESPONSE_FRACTION * strongest))
    ]
    found.sort(reverse=True)

    return [
        Hit(x_canon=x, y_canon=y, confidence=round(value / strongest, 3))
        for value, x, y in found[:MAX_HITS]
    ]


def _report(folder: Path, payload: dict, profile: TargetProfile, hole_diam_mm: float) -> None:
    """Measure one stored session: what the detector finds against what a person marked."""
    normalized = cv2.imread(str(folder / "normalized.png"))

    if normalized is None:
        raise SystemExit(f"no readable normalized.png in {folder}")

    truth = [(hit["x_canon"], hit["y_canon"]) for hit in payload["hits"]]

    try:
        proposal = detect(normalized, profile, hole_diam_mm)
    except (DetectorError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    result = agreement(
        truth,
        [(hit.x_canon, hit.y_canon) for hit in proposal],
        MATCH_TOL_MM / mm_per_px(profile),
    )
    offset = "—" if result.mean_offset_px is None else f"{result.mean_offset_px:.1f} px"

    print(f"{folder.name}: {len(truth)} marked by hand, {len(proposal)} proposed")
    print(f"  matched {result.matched} · missed {result.missed} · spurious {result.spurious}")
    print(f"  mean offset over the matched {offset}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="ta_client.cv_blob", description="Score the blob detector against a marked session."
    )
    parser.add_argument(
        "session", type=Path, help="a session folder: payload.json + normalized.png"
    )
    parser.add_argument("--hole-mm", type=float, default=DEFAULT_HOLE_DIAM_MM, metavar="MM")
    parser.add_argument(
        "--profile", type=Path, default=None, help="path to the target profile JSON"
    )
    args = parser.parse_args(argv)
    marked = args.session / "payload.json"

    if not marked.is_file():
        raise SystemExit(f"no payload.json in {args.session}")

    payload = json.loads(marked.read_text(encoding="utf-8"))
    profile_path = args.profile or (
        Path(__file__).resolve().parents[2]
        / "profiles"
        / f"{payload['session']['target_profile']}.json"
    )

    if not profile_path.is_file():
        raise SystemExit(f"no profile at {profile_path} — pass --profile")

    _report(args.session, payload, load_profile(profile_path), args.hole_mm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
