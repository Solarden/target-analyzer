"""Turn marked sessions into a YOLO dataset. See §13.

A hand-marked reading is a list of points; a detector is trained on boxes. The box is a
square of the calibre, because the canonical frame is metric and a hole's size is
therefore known rather than learned.

Tiles are cut by :mod:`ta_client.yolo`, not by a second implementation here. Training and
inference disagreeing about the grid by a few pixels would be invisible in both and wrong
in the result.

Every tile is written, including the ones holding no holes. An empty label file is a
background image, and that is how patch tape, printed digits, the paper edge and the
marker board are all represented without labelling anything.
"""

import argparse
import json
import shlex
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from ta_client.ship import _parts
from ta_client.yolo import IMGSZ, at_seam, geometry, origins, tiles
from ta_shared.payload import GROUND_TRUTH_METHOD
from ta_shared.profile import TargetProfile, hole_diam_px, load_profile


def _epoch(folder: Path) -> int:
    """When this folder was queued, for ordering two markings of one photo.

    Through the outbox's parser rather than a second one: the epoch is unpadded, so read as
    text "10-" lands before "2-". A folder renamed out of that shape sorts oldest, which
    only costs it a re-marking it was never going to win.
    """
    try:
        return _parts(folder).epoch
    except ValueError:
        return 0


def _readings(
    folders: list[Path], profile: TargetProfile
) -> dict[str, tuple[Path, list, np.ndarray]]:
    """The usable sessions, one per photo, keyed by the start of its hash.

    A photo re-marked later appears more than once; the newest folder wins, on the outbox's
    own rule. A session shot against another profile is skipped out loud — the tile size is
    in millimetres, so mixing profiles would be the one thing this cannot survive. Every
    image is read here rather than while writing, so a folder that cannot be used is
    refused before the dataset already on disk has been cleared.
    """
    # ponytail: holds every photo decoded, ~7 MB each. Read the shape here and decode in
    # the build loop instead once the corpus runs to hundreds.
    found: dict[str, tuple[Path, list, np.ndarray]] = {}
    canon = profile.canon_size_px

    for folder in sorted(folders, key=_epoch):
        marked = folder / "payload.json"

        if not marked.is_file() or not (folder / "normalized.png").is_file():
            print(f"{folder.name}: not a session folder, skipped", file=sys.stderr)

            continue

        payload = json.loads(marked.read_text(encoding="utf-8"))

        if payload["method"] != GROUND_TRUTH_METHOD:
            print(f"{folder.name}: {payload['method']} is a proposal, skipped", file=sys.stderr)

            continue

        if payload["session"]["target_profile"] != profile.name:
            other = payload["session"]["target_profile"]
            print(f"{folder.name}: shot against {other}, skipped", file=sys.stderr)

            continue

        digest = payload["image_sha256"][:12]

        # Same reason detector.py guards the profile name: a session folder can be written
        # by another machine, and this one names the tiles it will be cut into.
        if Path(digest).name != digest:
            print(f"{folder.name}: {digest!r} is not a photo hash, skipped", file=sys.stderr)

            continue

        normalized = cv2.imread(str(folder / "normalized.png"))

        if normalized is None or normalized.shape[:2] != (canon, canon):
            print(f"{folder.name}: not the profile's canonical square, skipped", file=sys.stderr)

            continue

        # Two markings of one photo can disagree by a few holes, so which one trains is a
        # choice worth seeing — the outbox says the same when it supersedes one.
        if digest in found:
            superseded = found[digest][0].name
            print(f"{superseded}: superseded by a later reading of the same photo", file=sys.stderr)

        found[digest] = (folder, payload["hits"], normalized)

    return found


def build(
    folders: list[Path], dest: Path, profile: TargetProfile, val: set[str]
) -> tuple[int, int]:
    """Write the dataset under ``dest``; return how many tiles and boxes it holds."""
    readings = _readings(folders, profile)

    if not readings:
        raise SystemExit("no usable marked session for this profile")

    # Naming a photo that is not here means a mistyped hash, and the split it was meant to
    # create would silently not exist — leaving data.yaml pointing at an empty val set.
    unknown = val - set(readings)

    if unknown:
        raise SystemExit(f"--val names no marked session: {', '.join(sorted(unknown))}")

    if set(readings) <= val:
        raise SystemExit("--val holds out every session, which leaves nothing to train on")

    # A data.yaml is what marks a directory as one of ours; anything else is somebody's.
    if dest.exists() and not (dest / "data.yaml").is_file() and any(dest.iterdir()):
        raise SystemExit(f"{dest} is not empty and holds no data.yaml — refusing to clear it")

    # Tiles left from an earlier split would put a photo in both train and val.
    for kind in ("images", "labels"):
        shutil.rmtree(dest / kind, ignore_errors=True)

    tile, stride, margin = geometry(profile)
    places = origins(profile.canon_size_px, tile, stride)
    side = hole_diam_px(profile) / tile
    canon = profile.canon_size_px
    written = boxes = 0

    for digest, (_folder, hits, normalized) in readings.items():
        split = "val" if digest in val else "train"

        for kind in ("images", "labels"):
            (dest / kind / split).mkdir(parents=True, exist_ok=True)

        for index, (ox, oy, crop) in enumerate(tiles(normalized, profile)):
            row, col = divmod(index, len(places))
            name = f"{digest}_r{row}c{col}"
            label = []

            for hit in hits:
                x, y = hit["x_canon"], hit["y_canon"]
                inside = ox <= x < ox + tile and oy <= y < oy + tile

                # at_seam is the rule inference applies, called rather than restated: a
                # hole the cut may have halved is left to the tile that holds it whole.
                if inside and not at_seam(x, y, ox, oy, tile, margin, canon):
                    label.append(
                        f"0 {(x - ox) / tile:.6f} {(y - oy) / tile:.6f} {side:.6f} {side:.6f}"
                    )

            cv2.imwrite(str(dest / "images" / split / f"{name}.png"), crop)
            (dest / "labels" / split / f"{name}.txt").write_text(
                "".join(f"{line}\n" for line in label), encoding="utf-8"
            )

            written += 1
            boxes += len(label)

    (dest / "data.yaml").write_text(
        # A JSON string is valid YAML, and quoting it is what keeps ": " or " #" in the
        # path from being read as syntax.
        f"path: {json.dumps(str(dest.resolve()))}\ntrain: images/train\n"
        f"val: images/{'val' if val else 'train'}\nnc: 1\nnames: [hole]\n",
        encoding="utf-8",
    )

    return written, boxes


def train_command(dest: Path) -> str:
    """The command that trains on what was just written.

    Printed rather than wrapped in a script so ``imgsz`` cannot drift from the tile the
    dataset was cut at, and so the augmentation below is visible instead of buried.

    Rotation and scale are off on purpose. The canonical frame fixes a hole's size, which
    is free information a scale jitter throws away, and a rotated square re-fits an
    axis-aligned box a third larger than the hole it marks. Flips are exact on such a box
    and cost nothing. Colour and exposure are the axis the corpus is genuinely poor in.
    """
    return (
        f"yolo detect train model=yolo11n.pt {shlex.quote(f'data={dest.resolve()}/data.yaml')} "
        f"imgsz={IMGSZ} epochs=300 patience=100 batch=16 device=mps "
        "single_cls=True seed=0 cache=True "
        "degrees=0 scale=0 mosaic=0 mixup=0 copy_paste=0 "
        "fliplr=0.5 flipud=0.5 translate=0.05 hsv_s=0.5 hsv_v=0.5"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="ta_client.yolo_data", description="Build a YOLO dataset from marked sessions."
    )
    parser.add_argument("sessions", type=Path, nargs="+", help="session folders")
    parser.add_argument("--out", type=Path, required=True, help="where to write the dataset")
    parser.add_argument("--profile", type=Path, required=True, help="the target profile JSON")
    parser.add_argument(
        "--val", action="append", default=[], metavar="SHA12", help="hold this photo out"
    )
    args = parser.parse_args(argv)
    profile = load_profile(args.profile)
    written, boxes = build(args.sessions, args.out, profile, set(args.val))

    print(f"{written} tiles, {boxes} boxes -> {args.out}")
    print(train_command(args.out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
