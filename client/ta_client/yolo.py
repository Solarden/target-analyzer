"""Hole detection with a model trained on this project's own photos. See §13.

The frame is read in tiles rather than whole. A hole is 13 px in a 1500 px canonical
square, and a detector that resizes the square to its own input turns that into five —
below the grid the box centre is regressed on, which is the one quantity this detector
exists to place better than a blob filter.

Tiles are measured in **millimetres**, not pixels. The canonical square spans a different
physical target per profile, so a fixed pixel tile would show the model a 13 px hole on
one profile and a 36 px hole on another. A fixed millimetre tile, letterboxed to
``IMGSZ``, shows it the same hole every time.

The model itself is reached through ``predict``, injected the way ``vlm``'s sender is, so
everything here except the weights can be exercised with neither a model nor torch.
"""

import argparse
import sys
from collections.abc import Callable
from functools import lru_cache
from math import hypot
from pathlib import Path

import numpy as np

from ta_client.config import Settings, get_settings
from ta_client.detector import DetectorError, load_session, report
from ta_shared.agreement import MATCH_TOL_MM
from ta_shared.payload import MAX_HITS, Hit
from ta_shared.profile import TargetProfile, hole_diam_px, mm_per_px

TILE_MM = 100.0
IMGSZ = 640
# Wide enough that a hole cut by one tile's edge sits well inside its neighbour.
OVERLAP_MM = 20.0
# Sixty-four tiles of IMGSZ square in one tensor is most of a gigabyte for no gain.
BATCH = 16

# x1, y1, x2, y2, confidence — in the pixels of the tile handed to the predictor.
Box = tuple[float, float, float, float, float]
Predictor = Callable[[list[np.ndarray], Settings], list[list[Box]]]


class YoloError(DetectorError):
    """This model cannot read the photo now: not installed, not loadable, not there."""


def origins(canon: int, tile: int, stride: int) -> list[int]:
    """Tile offsets along one axis, the last flush with the far edge.

    Clamping the last one rather than striding past it keeps every tile inside the frame
    and costs only a wider overlap at the end.
    """
    if tile >= canon:
        return [0]

    return [*range(0, canon - tile, stride), canon - tile]


def geometry(profile: TargetProfile) -> tuple[int, int, int]:
    """Tile side, stride and seam margin, all in canonical pixels."""
    scale = mm_per_px(profile)

    if scale is None:
        raise ValueError("the profile has no target_diam_mm, so a tile has no size in pixels")

    tile = min(round(TILE_MM / scale), profile.canon_size_px)

    return tile, tile - round(OVERLAP_MM / scale), round(hole_diam_px(profile))


def tiles(normalized: np.ndarray, profile: TargetProfile) -> list[tuple[int, int, np.ndarray]]:
    """The frame as overlapping crops, each with the canonical offset of its top-left."""
    tile, stride, _margin = geometry(profile)
    places = origins(profile.canon_size_px, tile, stride)

    return [(x, y, normalized[y : y + tile, x : x + tile]) for y in places for x in places]


def at_seam(x: float, y: float, ox: int, oy: int, tile: int, margin: int, canon: int) -> bool:
    """Is this detection close enough to a cut edge that the hole may be half off it?

    The frame's own edges do not count: nothing was cut there, and a hole near them has
    no second tile to be found in.
    """
    for value, origin in ((x, ox), (y, oy)):
        if origin > 0 and value - origin < margin:
            return True

        if origin + tile < canon and origin + tile - value < margin:
            return True

    return False


def _merge(found: list[tuple[float, float, float]], profile: TargetProfile) -> list[Hit]:
    """Strongest first, dropping whatever a stronger detection already accounts for.

    The radius is the tolerance that defines "the same hole" everywhere else in the
    project, so a duplicate across an overlap is suppressed by the same rule that would
    later score the two as one. Distance rather than box overlap, because averaging two
    boxes moves the centre — which is the number this detector is judged on.
    """
    # ponytail: O(n²) over at most MAX_HITS points. A grid index if a photo ever carries
    # hundreds of holes, which one string does not.
    radius = MATCH_TOL_MM / mm_per_px(profile)
    kept: list[tuple[float, float, float]] = []

    for confidence, x, y in sorted(found, reverse=True):
        if all(hypot(x - kx, y - ky) > radius for _c, kx, ky in kept):
            kept.append((confidence, x, y))

    return [
        Hit(x_canon=x, y_canon=y, confidence=round(confidence, 3))
        for confidence, x, y in kept[:MAX_HITS]
    ]


@lru_cache(maxsize=1)
def _model(weights: str, device: str):
    """The loaded model, kept so a sweep over stored sessions loads it once.

    Keyed on two strings rather than on ``Settings``, which is a mutable pydantic model
    and so unhashable. The ultralytics import lives in here because importing this module
    must cost nothing on a machine that has no model and no torch.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise YoloError("ultralytics is not installed — run `uv sync --extra yolo`") from exc

    try:
        model = YOLO(weights)

        if device:
            model.to(device)
    except Exception as exc:
        raise YoloError(f"could not load the model: {exc}") from exc

    return model


def run(crops: list[np.ndarray], settings: Settings) -> list[list[Box]]:
    """Ask the model about each tile. An empty batch loads it and asks nothing."""
    model = _model(str(settings.yolo_weights), settings.yolo_device)
    boxes: list[list[Box]] = []

    for start in range(0, len(crops), BATCH):
        results = model.predict(
            crops[start : start + BATCH], imgsz=IMGSZ, conf=settings.yolo_conf, verbose=False
        )

        for result in results:
            xyxy = result.boxes.xyxy.tolist()
            confidences = result.boxes.conf.tolist()
            boxes.append([(*box, float(c)) for box, c in zip(xyxy, confidences, strict=True)])

    return boxes


def model_name(settings: Settings | None = None) -> str | None:
    """Which model produced the reading — the weights file, as its owner named it."""
    weights = (settings or get_settings()).yolo_weights

    return weights.stem if weights else None


def check(settings: Settings | None, profile: TargetProfile, *, predict: Predictor = run) -> None:
    """Raise what ``detect`` would raise, early enough that no clicks are lost to it."""
    settings = settings or get_settings()
    weights = settings.yolo_weights

    if weights is None:
        raise YoloError("TA_YOLO_WEIGHTS is not set — point it at a trained model")

    if not weights.is_file():
        raise YoloError(f"no model file at {weights}")

    # Every ultralytics run writes weights/best.pt, so the name identifies nothing. It is
    # stored as the reading's provenance and cannot be re-derived from the hits later.
    if weights.stem in ("best", "last"):
        raise YoloError(f"{weights.name} names the training run, not the model — rename it")

    if mm_per_px(profile) is None:
        raise YoloError(
            f"{profile.name} v{profile.version} has no target_diam_mm, so a tile has no "
            "size in canonical pixels"
        )

    # Loading is the slow, fallible half and all of it is knowable now: the import, the
    # file and the device. An empty batch proves them and asks the model nothing.
    predict([], settings)


def detect(
    normalized: np.ndarray,
    profile: TargetProfile,
    *,
    settings: Settings | None = None,
    predict: Predictor = run,
) -> list[Hit]:
    """Propose holes in a canonical-frame image, strongest first."""
    settings = settings or get_settings()
    canon = profile.canon_size_px

    if normalized.ndim != 3 or normalized.shape[:2] != (canon, canon):
        shape = "x".join(str(n) for n in normalized.shape)

        raise ValueError(
            f"image is {shape}, not {profile.name} v{profile.version}'s "
            f"{canon}x{canon}x3 canonical square"
        )

    check(settings, profile, predict=predict)
    tile, _stride, margin = geometry(profile)
    crops = tiles(normalized, profile)
    reading = predict([crop for _x, _y, crop in crops], settings)
    found = []

    for (ox, oy, _crop), boxes in zip(crops, reading, strict=True):
        for x1, y1, x2, y2, confidence in boxes:
            x = ox + (x1 + x2) / 2
            y = oy + (y1 + y2) / 2

            if not at_seam(x, y, ox, oy, tile, margin, canon):
                found.append((confidence, x, y))

    return _merge(found, profile)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="ta_client.yolo", description="Score the trained detector against a marked session."
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
