"""Write every hand-marked reading back out as a session folder.

    python -m target_analyzer.export_corpus corpus/sessions

Run where the database and ``TA_DATA_PATH`` already resolve. The outbox deletes each
folder the moment the server accepts it, so the Mac keeps nothing and this is the only
way back to the photos a detector was ever measured on.

The output is the ordinary session layout — ``payload.json`` + ``normalized.png`` — and
not a corpus format of its own, so `ta_client.detector.load_session` reads an export and
a live outbox folder identically. Only the ground truth is exported: a detector's reading
is a proposal, and nothing should train or measure against one.
"""

import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session, select

from ta_shared.payload import GROUND_TRUTH_METHOD, Hit, SessionMeta, ShipPayload
from target_analyzer.config import get_settings
from target_analyzer.db import get_engine
from target_analyzer.models import Hole, Image, Interpretation, ShootingSession, TargetProfile


def _utc(moment: datetime) -> datetime:
    """A stored instant, with the zone it was written in put back.

    Everything persisted here is UTC, but neither backend stores the offset, so a value
    read back is naive — and a naive datetime reports its epoch as if it were local time.
    """
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def export(session: Session, base: Path, dest: Path) -> int:
    """Write each hand-marked reading under ``dest``; return how many were written."""
    rows = session.exec(
        select(Interpretation, Image, ShootingSession, TargetProfile)
        .join(Image, Interpretation.image_id == Image.id)
        .join(ShootingSession, Image.session_id == ShootingSession.id)
        .join(TargetProfile, Interpretation.target_profile_id == TargetProfile.id)
        .where(Interpretation.method == GROUND_TRUTH_METHOD)
    ).all()
    holes: dict[int, list[Hole]] = {}

    for hole in session.exec(
        select(Hole)
        .join(Interpretation, Hole.interpretation_id == Interpretation.id)
        .where(Interpretation.method == GROUND_TRUTH_METHOD)
        .order_by(Hole.id)
    ):
        holes.setdefault(hole.interpretation_id, []).append(hole)

    written = 0

    for interpretation, image, shooting, profile in rows:
        source = base / image.normalized_path

        # A row whose file is gone is worth naming and stepping over: an export that dies
        # halfway through a partially restored data directory helps nobody.
        if not source.is_file():
            print(f"{image.sha256[:12]}: no normalized image on disk, skipped", file=sys.stderr)

            continue

        payload = ShipPayload(
            image_sha256=image.sha256,
            canon_size_px=image.canon_size_px,
            method=GROUND_TRUTH_METHOD,
            model=interpretation.model,
            params=interpretation.params,
            session=SessionMeta(
                gun=shooting.gun,
                distance_m=shooting.distance_m,
                notes=shooting.notes,
                shot_at=shooting.shot_at,
                target_profile=profile.name,
                target_profile_version=profile.version,
            ),
            hits=[
                Hit(x_canon=h.x_canon, y_canon=h.y_canon, confidence=h.confidence)
                for h in holes.get(interpretation.id, [])
            ],
            created_at=_utc(interpretation.created_at),
        )
        epoch = int(_utc(image.uploaded_at).timestamp())
        folder = dest / f"{epoch}-{image.sha256[:12]}-{GROUND_TRUTH_METHOD}"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, folder / "normalized.png")
        # Last, because payload.json is what marks a folder as readable.
        (folder / "payload.json").write_text(payload.model_dump_json(), encoding="utf-8")
        written += 1

    return written


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m target_analyzer.export_corpus <dir>", file=sys.stderr)

        return 2

    dest = Path(argv[0])

    with Session(get_engine()) as session:
        written = export(session, get_settings().data_path, dest)

    print(f"exported {written} hand-marked session(s) to {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
