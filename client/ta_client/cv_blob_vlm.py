"""Blob detection, vetted by a vision model. See implementation.md §9 and §13.

The two detectors this is built from fail in opposite directions: `cv_blob` places a hole
to about a millimetre but cannot tell one from a printed digit, and `vlm` can tell them
apart but barely finds them.

Resolution is why, and it is the whole design. A hole is about 13 px across in the 1500 px
canonical frame, and a vision encoder resamples that to a handful of pixels; the same hole
in a crop reaches the model hundreds of pixels wide. So the blob filter proposes, the model
is asked about each proposal on its own, and the survivors keep the blob filter's
coordinates — the judgement is replaced, the placement is not.

The cost is one round trip per candidate rather than one per photo, which is why they share
a connection. A model that fails partway fails the whole reading on purpose: a filter that
silently dropped the candidates it could not ask about would report a short list as though
it were a complete one.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ta_client import cv_blob, vlm
from ta_client.config import Settings, get_settings
from ta_client.detector import load_session, report
from ta_shared.payload import MAX_HITS, Hit
from ta_shared.profile import TargetProfile, mm_per_px

# Wide enough that a printed ring line or digit is recognisable as one, tight enough that
# the candidate is unambiguously the thing in the middle.
CROP_HOLE_WIDTHS = 7.5
# The side the crop is sent at. Vision encoders work at a few hundred pixels, so past this
# the upscale is carrying no more information into the model.
CROP_SEND_PX = 448
# What one warm crop is allowed to cost. TA_VLM_TIMEOUT covers the cold load, which is
# paid once, and this covers each candidate after it — without a whole-run budget a model
# that answers slowly rather than failing holds the hole picker shut for an hour.
WARM_SECONDS = 20.0

PROMPT = """This is a close-up crop of a paper shooting target. Look at the object at the
very centre of the image. Is it a bullet hole torn through the paper, or is it something
else — a printed digit, a printed ring line, or white patch tape covering an older hole?

Answer with JSON and nothing else: {"answer": "hole"} or {"answer": "not"}"""


def model_name(settings: Settings | None = None) -> str | None:
    """Which model vetted the reading."""
    return vlm.model_name(settings)


def check(settings: Settings | None, profile: TargetProfile) -> None:
    """Refuse a run that either half cannot complete, before any clicking is done."""
    cv_blob.check(settings, profile)
    vlm.check(settings, profile)


def crop(padded: np.ndarray, x: float, y: float, half: int, canon: int) -> np.ndarray:
    """One candidate's window, from a canonical frame padded by ``half`` on every side.

    The padding is checked rather than assumed: handed the unpadded frame this would
    still slice successfully, and every window would be centred half a crop away.
    """
    side = canon + 2 * half

    if padded.shape[:2] != (side, side):
        raise ValueError(f"expected a frame padded to {side}x{side}, got {padded.shape[:2]}")

    left, top = round(x), round(y)

    return padded[top : top + 2 * half, left : left + 2 * half]


def verdict(answer: str) -> bool:
    """Whether the model called the crop's centre a hole."""
    try:
        value = vlm.first_json_object(answer)["answer"]
    except (KeyError, TypeError) as exc:
        raise vlm.VlmError(f"answer is not an object with an 'answer' key: {exc}") from exc

    if value not in ("hole", "not"):
        raise vlm.VlmError(f"{value!r} is neither 'hole' nor 'not'")

    return value == "hole"


def detect(
    normalized: np.ndarray,
    profile: TargetProfile,
    *,
    settings: Settings | None = None,
    send: vlm.Sender = vlm.post,
) -> list[Hit]:
    """Propose holes with the blob filter, then keep the ones a vision model calls holes.

    Raises ``ValueError`` when the image is not the profile's canonical square, and
    ``VlmError`` when the model cannot be reached or does not answer with a verdict.
    """
    settings = settings or get_settings()
    cv_blob.check(settings, profile)
    # Before the endpoint is checked, so a frame that is not the canonical square raises
    # ValueError here rather than being reported as a configuration problem.
    candidates = cv_blob.detect(normalized, profile)
    vlm.check(settings, profile)
    canon = profile.canon_size_px
    half = round(CROP_HOLE_WIDTHS * cv_blob.DEFAULT_HOLE_DIAM_MM / mm_per_px(profile) / 2)
    # Replicated rather than black: a candidate near the paper's edge stays in the middle
    # of its crop, where the prompt says it is, instead of against a border it invented.
    padded = cv2.copyMakeBorder(normalized, half, half, half, half, cv2.BORDER_REPLICATE)
    deadline = time.monotonic() + settings.vlm_timeout + WARM_SECONDS * len(candidates)
    kept, unreadable = [], 0

    with vlm.client_for(settings) as client:
        for index, candidate in enumerate(candidates):
            if time.monotonic() > deadline:
                raise vlm.VlmError(
                    f"gave up after {index} of {len(candidates)} candidates: the model is "
                    "answering, but too slowly to finish this photo"
                )

            window = crop(padded, candidate.x_canon, candidate.y_canon, half, canon)
            image = cv2.resize(window, (CROP_SEND_PX, CROP_SEND_PX), interpolation=cv2.INTER_CUBIC)
            answer = vlm.ask(client, vlm.encode(image), PROMPT, settings, send)

            # Kept rather than dropped, and unlike an unreachable model this does not end
            # the reading: one hedged verdict costs a proposal to delete, where failing
            # here would discard every candidate already judged.
            try:
                accepted = verdict(answer)
            except vlm.VlmError:
                accepted, unreadable = True, unreadable + 1

            if accepted:
                kept.append(candidate)

    note = f", {unreadable} unreadable and kept" if unreadable else ""
    print(f"cv_blob_vlm: kept {len(kept)} of {len(candidates)} proposals{note}", file=sys.stderr)

    return kept[:MAX_HITS]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="ta_client.cv_blob_vlm",
        description="Score the model-vetted blob detector against a marked session.",
    )
    parser.add_argument(
        "session", type=Path, help="a session folder: payload.json + normalized.png"
    )
    parser.add_argument(
        "--profile", type=Path, default=None, help="path to the target profile JSON"
    )
    args = parser.parse_args(argv)
    payload, profile = load_session(args.session, args.profile)
    report(args.session, payload, profile, detect)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
