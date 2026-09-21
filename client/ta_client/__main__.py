"""One photo, end to end: ``python -m ta_client <photo> --gun … --distance …``.
See implementation.md §9.

Load, register, mark the holes, package, ship. A session the server cannot take stays
in the outbox and goes out on the next run.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import cv2
from pydantic import ValidationError

from ta_client import cv_blob, ship, vlm
from ta_client.config import get_settings
from ta_client.detector import DetectorError
from ta_client.hits import pick_hits
from ta_client.load import load_stripped
from ta_client.package import build_payload
from ta_client.register import register_interactive
from ta_shared.payload import GROUND_TRUTH_METHOD, SessionMeta
from ta_shared.profile import load_profile

# The detectors this client can run. --method's choices and the call that runs one both
# read it, so every name the flag accepts is a name that does something. Modules rather
# than functions, because a reading is three things the runner needs and one of them —
# which model answered — is not in the hits.
DETECTORS = {"cv_blob": cv_blob, "vlm": vlm}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ta_client", description=__doc__.splitlines()[0])
    parser.add_argument("photo", type=Path)
    parser.add_argument("--gun", required=True)
    parser.add_argument("--distance", type=float, required=True, metavar="METRES")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=date.today(),
        metavar="YYYY-MM-DD",
        help="the day it was shot (default: today) — EXIF is stripped, so it cannot be derived",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--method",
        choices=(GROUND_TRUTH_METHOD, *DETECTORS),
        default=GROUND_TRUTH_METHOD,
        help="run this detector first and correct what it proposes (default: manual)",
    )
    # No default: the profile is the scoring geometry, and picking the wrong one produces
    # a complete, plausible session with every hit in the wrong ring and nothing to say so.
    parser.add_argument(
        "--profile", type=Path, required=True, help="path to the target profile JSON"
    )
    # Not --out: ta_client.board already spends that name on an output SVG file.
    parser.add_argument(
        "--outbox", type=Path, default=None, metavar="DIR", help="the outbox to write into"
    )
    args = parser.parse_args(argv)

    if not args.profile.is_file():
        parser.error(f"no profile at {args.profile} — run from the repo root, or pass --profile")

    settings = get_settings()
    profile = load_profile(args.profile)

    # Built before anything interactive: a bad --distance or an over-long --gun is a
    # pydantic error, and raising it after a session has been clicked throws that work away.
    try:
        session = SessionMeta(
            gun=args.gun,
            distance_m=args.distance,
            notes=args.notes,
            shot_at=args.date,
            target_profile=profile.name,
            target_profile_version=profile.version,
        )
    except ValidationError as exc:
        # Field names, not flag names: --distance carries distance_m, and inventing a flag
        # that does not exist would be worse than making the reader map one to the other.
        parser.error("; ".join(f"{e['loc'][0]}: {e['msg']}" for e in exc.errors()))

    # Also before anything interactive: a missing token is a config error, and discovering
    # it after forty holes have been clicked is the same wasted work as a bad --distance.
    if settings.server_url:
        try:
            ship.check_config(settings)
        except ship.ShipError as exc:
            parser.error(str(exc))

    detector = DETECTORS.get(args.method)

    # Same reason: whatever a detector needs — a physical scale, an endpoint to ask —
    # raising it after the photo has been registered by hand throws the registration away.
    if detector is not None:
        try:
            detector.check(settings, profile)
        except DetectorError as exc:
            parser.error(str(exc))

    photo, original_jpg, digest = load_stripped(args.photo)
    registration = register_interactive(photo, profile)

    if registration is None:
        print("registration cancelled", file=sys.stderr)

        return 1

    proposal = None

    if detector is not None:
        try:
            proposal = detector.detect(registration.normalized, profile)
            print(f"{args.method} proposes {len(proposal)} holes", file=sys.stderr)
        except DetectorError as exc:
            # The registration is already spent by the time this can fail.
            print(f"{args.method} could not read it: {exc}", file=sys.stderr)
            print("marking by hand — only the second reading is lost", file=sys.stderr)

    hits = pick_hits(registration.normalized, profile, proposal)

    if hits is None:
        print("marking cancelled", file=sys.stderr)

        return 1

    # Both readings of the one photo: corrected in place, the row would measure the
    # person rather than the detector, and leave nothing to compare it against.
    readings = [(GROUND_TRUTH_METHOD, hits, None)]

    if proposal is not None:
        readings.append((args.method, proposal, detector.model_name(settings)))

    ok, buffer = cv2.imencode(".png", registration.normalized)

    if not ok:
        raise ValueError("could not encode the normalized image")

    outbox = args.outbox or settings.outbox_path
    print(f"{len(hits)} hits · {registration.summary(profile)}")

    for position, (method, reading, model) in enumerate(readings):
        payload = build_payload(
            image_sha256=digest,
            profile=profile,
            registration=registration,
            hits=reading,
            session=session,
            method=method,
            model=model,
        )
        # Only with the reading the outbox sends first, which is the one that creates the
        # image row. The server re-hashes a second copy and then has nowhere to put it.
        original = original_jpg if settings.ship_original and position == 0 else None
        destination = ship.enqueue(outbox, payload, buffer.tobytes(), original)
        print(f"{digest[:12]} {method} -> {destination}")

    if not settings.server_url:
        print("TA_SERVER_URL is not set — the session stays queued", file=sys.stderr)

        return 0

    try:
        ship.flush(outbox, settings=settings)
    except ship.ShipError as exc:
        print(exc, file=sys.stderr)

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
