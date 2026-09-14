"""Score and precision over time, filtered by gun and distance (implementation.md §10)."""

import math

from fastapi import Request
from fastapi.responses import HTMLResponse

from target_analyzer.api.deps import CurrentUser, DbSession
from target_analyzer.api.endpoints.dashboard import dashboard_router
from target_analyzer.queries import trend
from target_analyzer.templating import templates

router = dashboard_router()


def _opt_distance(value: str) -> float | None:
    """The distance filter as a finite float, or None when the param is not one.

    The filter bar auto-submits raw strings and a URL can be hand-edited, so a bad value
    falls back to "no filter" rather than 422-ing. The finite check earns its place:
    ``float()`` accepts "nan" and "inf", either of which matches no session at all.
    """
    if not value:
        return None

    try:
        parsed = float(value)
    except ValueError:
        return None

    return parsed if math.isfinite(parsed) else None


def _chart(rows: list[trend.TrendRow]) -> dict:
    """Labels and series for the chart, as plain JSON-able data.

    Every series emits ``None`` rather than skipping a point: a profile with no
    physical diameter reports no millimetres, and a shorter array would silently
    slide that series along the x-axis.
    """
    return {
        # The shooting day when it is known, not the ingest timestamp: a batch of old
        # photos processed in one evening would otherwise collapse onto today.
        "labels": [
            (shooting.shot_at or interpretation.created_at.date()).isoformat()
            for interpretation, shooting in rows
        ],
        "score": [interpretation.total_score for interpretation, _ in rows],
        "mean_radius_mm": [interpretation.mean_radius_mm for interpretation, _ in rows],
        "extreme_spread_mm": [interpretation.extreme_spread_mm for interpretation, _ in rows],
        "bias_mm": [interpretation.bias_mm for interpretation, _ in rows],
    }


@router.get("", response_class=HTMLResponse)
def trend_view(
    request: Request,
    session: DbSession,
    user: CurrentUser,
    gun: str = "",
    distance: str = "",
) -> HTMLResponse:
    distance_m = _opt_distance(distance)
    rows = trend.rows(session, gun=gun or None, distance_m=distance_m)

    return templates.TemplateResponse(
        request,
        "dashboard/trend.html",
        {
            "user": user,
            "rows": rows,
            "chart": _chart(rows),
            "guns": trend.guns(session),
            "distances": trend.distances(session),
            # The parsed distance, not the raw one: `?distance=25` filters fine but
            # never equals the option's "25.0", so the select would read "All".
            "f_gun": gun,
            "f_distance": str(distance_m) if distance_m is not None else "",
        },
    )
