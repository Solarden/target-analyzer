"""The Trend view's reads: one row per interpretation, plus its filter options."""

from sqlmodel import Session, col, distinct, select

from target_analyzer.models import Image, Interpretation, ShootingSession

TrendRow = tuple[Interpretation, ShootingSession]


def rows(
    session: Session, *, gun: str | None = None, distance_m: float | None = None
) -> list[TrendRow]:
    """Every interpretation with the session it belongs to, oldest first.

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
        # id breaks the tie: two readings of one photo share a transaction and can
        # share a timestamp, and an unstable order reshuffles the chart on refresh.
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
