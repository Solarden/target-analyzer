"""The Trend view's reads: one row per confirmed reading, plus its filter options."""

from sqlmodel import Session, col, distinct, select

from ta_shared.payload import GROUND_TRUTH_METHOD
from target_analyzer.models import Image, Interpretation, ShootingSession

TrendRow = tuple[Interpretation, ShootingSession]


def rows(
    session: Session, *, gun: str | None = None, distance_m: float | None = None
) -> list[TrendRow]:
    """Every hand-marked reading with the session it belongs to, oldest first.

    Hand-marked only: this chart is how the shooting is going, and a detector's reading
    of a photo is a second opinion on one target rather than a second string. Counting
    both draws a climb or a slump out of one session, which is the opposite of what a
    trend is for. Compare is where a detector is read.

    Chronological because the chart reads left to right and the table beneath it
    shows the same rows in the same order — two orders would make the table look
    like a different dataset.
    """
    # ponytail: whole rows, so each drags its params and metrics JSON for columns Trend
    # never reads. Name the columns if the page ever gets slow.
    statement = (
        select(Interpretation, ShootingSession)
        # Image is joined to reach the session but never selected — Trend renders
        # nothing from it.
        .join(Image, col(Interpretation.image_id) == col(Image.id))
        .join(ShootingSession, col(Image.session_id) == col(ShootingSession.id))
        .where(Interpretation.method == GROUND_TRUTH_METHOD)
        # id breaks the tie: an outbox flush ingests a backlog in one go, so several
        # sessions can share a timestamp, and an unstable order reshuffles the chart.
        .order_by(col(Interpretation.created_at), col(Interpretation.id))
    )

    if gun:
        statement = statement.where(ShootingSession.gun == gun)

    if distance_m is not None:
        statement = statement.where(ShootingSession.distance_m == distance_m)

    return list(session.exec(statement).all())


def guns(session: Session) -> list[str]:
    statement = select(distinct(col(ShootingSession.gun))).order_by(col(ShootingSession.gun))

    return list(session.exec(statement).all())


def distances(session: Session) -> list[float]:
    statement = select(distinct(col(ShootingSession.distance_m))).order_by(
        col(ShootingSession.distance_m)
    )

    return list(session.exec(statement).all())
