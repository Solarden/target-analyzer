"""The Session and Compare views' reads.

Both views render the same thing — an image with one reading of it drawn on top —
so both build :class:`Overlay`. Session shows one per photo; Compare shows every
reading of a single photo side by side.
"""

from dataclasses import dataclass

from sqlmodel import Session, col, select

from ta_shared import payload
from target_analyzer.models import Hole, Image, Interpretation, ShootingSession, TargetProfile

# The reading Session draws; Compare is where the others earn their place (§13).
GROUND_TRUTH_METHOD = payload.GROUND_TRUTH_METHOD


@dataclass(frozen=True)
class Overlay:
    """One reading of one image, plus everything the target overlay needs to draw it.

    ``profile`` is the geometry this interpretation actually scored against, not the
    session's current one — profiles are versioned and a re-measure must not move
    the rings under a reading that predates it.
    """

    image: Image
    interpretation: Interpretation
    profile: TargetProfile
    holes: list[Hole]

    @property
    def confirmed(self) -> bool:
        """Whether a person marked this reading, as opposed to a detector proposing it."""
        return self.interpretation.method == GROUND_TRUTH_METHOD


def _build(session: Session, pairs: list[tuple[Image, Interpretation]]) -> list[Overlay]:
    """Attach holes and profiles to (image, interpretation) pairs in two queries."""
    if not pairs:
        return []

    interpretation_ids = [interpretation.id for _, interpretation in pairs]
    holes = session.exec(
        select(Hole)
        .where(col(Hole.interpretation_id).in_(interpretation_ids))
        .order_by(col(Hole.id))
    ).all()
    by_interpretation: dict[int, list[Hole]] = {}

    for hole in holes:
        by_interpretation.setdefault(hole.interpretation_id, []).append(hole)

    profile_ids = {interpretation.target_profile_id for _, interpretation in pairs}
    profiles = session.exec(
        select(TargetProfile).where(col(TargetProfile.id).in_(profile_ids))
    ).all()
    by_profile = {profile.id: profile for profile in profiles}

    return [
        Overlay(
            image=image,
            interpretation=interpretation,
            profile=by_profile[interpretation.target_profile_id],
            holes=by_interpretation.get(interpretation.id, []),
        )
        for image, interpretation in pairs
    ]


def for_session(session: Session, session_id: int) -> tuple[ShootingSession, list[Overlay]] | None:
    """A shooting session with one overlay per photo, or None if there is no such session."""
    shooting_session = session.get(ShootingSession, session_id)

    if shooting_session is None:
        return None

    images = session.exec(
        select(Image).where(Image.session_id == session_id).order_by(col(Image.id))
    ).all()

    if not images:
        return shooting_session, []

    interpretations = session.exec(
        select(Interpretation)
        .where(col(Interpretation.image_id).in_([image.id for image in images]))
        .order_by(col(Interpretation.id))
    ).all()
    pairs = []

    for image in images:
        candidates = [item for item in interpretations if item.image_id == image.id]

        if not candidates:
            continue

        chosen = next(
            (item for item in candidates if item.method == GROUND_TRUTH_METHOD), candidates[0]
        )
        pairs.append((image, chosen))

    return shooting_session, _build(session, pairs)


def for_image(
    session: Session, image_id: int
) -> tuple[Image, ShootingSession, list[Overlay]] | None:
    """One photo with every reading of it, or None if there is no such image."""
    image = session.get(Image, image_id)

    if image is None:
        return None

    shooting_session = session.get(ShootingSession, image.session_id)

    if shooting_session is None:
        return None

    interpretations = session.exec(
        select(Interpretation)
        .where(Interpretation.image_id == image_id)
        # Ground truth first, then alphabetically: the baseline every other method is
        # read against belongs in the leftmost column, not wherever its name sorts.
        .order_by(col(Interpretation.method) != GROUND_TRUTH_METHOD, col(Interpretation.method))
    ).all()

    return image, shooting_session, _build(session, [(image, item) for item in interpretations])
