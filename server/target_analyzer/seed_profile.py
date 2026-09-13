"""Insert a target profile from its JSON artifact into the database.

    python -m target_analyzer.seed_profile profiles/issf_precision.json

Profiles are immutable and versioned (§4): re-measuring the physical target edits the
JSON, bumps ``version``, and seeds again — this never UPDATEs, so every past
interpretation keeps pointing at the geometry it was actually scored against.

Loading through :func:`ta_shared.profile.load_profile` rather than hand-writing SQL is
the point: it validates the ring geometry before anything reaches a column, and the
``board`` JSON is a shape that is easy to get subtly wrong by hand.
"""

import sys
from pathlib import Path

from sqlmodel import Session, select

from ta_shared.profile import load_profile
from target_analyzer.db import get_engine
from target_analyzer.models import TargetProfile


def seed(session: Session, path: Path) -> tuple[TargetProfile, bool]:
    """Insert the profile at ``path``; return it and whether it was newly created."""
    profile = load_profile(path)

    existing = session.exec(
        select(TargetProfile).where(
            TargetProfile.name == profile.name, TargetProfile.version == profile.version
        )
    ).first()

    if existing is not None:
        return existing, False

    row = TargetProfile(
        name=profile.name,
        version=profile.version,
        n_rings=profile.n_rings,
        canon_size_px=profile.canon_size_px,
        ring_radii_px=profile.ring_radii_px,
        target_diam_mm=profile.target_diam_mm,
        board=profile.board.model_dump(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)

    return row, True


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m target_analyzer.seed_profile <profile.json>", file=sys.stderr)

        return 2

    with Session(get_engine()) as session:
        row, created = seed(session, Path(argv[0]))

    verb = "seeded" if created else "already present"
    print(f"{verb}: {row.name} v{row.version} (id={row.id}, canon={row.canon_size_px}px)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
