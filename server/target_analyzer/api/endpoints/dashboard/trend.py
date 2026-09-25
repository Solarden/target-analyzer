"""Score and precision over time, filtered by gun, distance and shooter (implementation.md §10)."""

import math

from fastapi import Request
from fastapi.responses import HTMLResponse

from target_analyzer.api.deps import CurrentUser, DbSession
from target_analyzer.api.endpoints.dashboard import dashboard_router
from target_analyzer.queries import trend
from target_analyzer.templating import format_shooter, templates

router = dashboard_router()

# The chart's choices: the query value, what the axis reads, and the column it plots.
METRICS = {
    "score": ("Total score", "total_score"),
    "mean_radius_mm": ("Mean radius (mm)", "mean_radius_mm"),
    "extreme_spread_mm": ("Extreme spread (mm)", "extreme_spread_mm"),
    "bias_mm": ("Bias (mm)", "bias_mm"),
}


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


def _chart(rows: list[trend.TrendRow], metric: str, everyone: list[str | None]) -> dict:
    """Labels and one series per shooter for one metric, as plain JSON-able data.

    Every series is as long as the labels, with ``None`` where the row is someone else's
    or the metric is missing — a profile with no physical diameter reports no millimetres
    — since a shorter array would slide that series along the x-axis.
    """
    title, column = METRICS[metric]
    present = {shooting.shooter for _, shooting in rows}

    return {
        # The shooting day when it is known, not the ingest timestamp: a batch of old
        # photos processed in one evening would otherwise collapse onto today.
        "labels": [
            (shooting.shot_at or interpretation.created_at.date()).isoformat()
            for interpretation, shooting in rows
        ],
        "title": title,
        "series": [
            {
                "label": format_shooter(name),
                # A place among everyone who has shot, so a shooter keeps one colour
                # under every filter.
                "colour": place,
                "data": [
                    getattr(interpretation, column) if shooting.shooter == name else None
                    for interpretation, shooting in rows
                ],
            }
            for place, name in enumerate(everyone)
            if name in present
        ],
    }


@router.get("", response_class=HTMLResponse)
def trend_view(
    request: Request,
    session: DbSession,
    user: CurrentUser,
    gun: str = "",
    distance: str = "",
    shooter: str = "",
    metric: str = "score",
) -> HTMLResponse:
    distance_m = _opt_distance(distance)
    # Like a bad distance, a hand-edited metric falls back rather than 422-ing.
    metric = metric if metric in METRICS else "score"
    rows = trend.rows(session, gun=gun or None, distance_m=distance_m, shooter=shooter or None)
    shooters = trend.shooters(session)

    return templates.TemplateResponse(
        request,
        "dashboard/trend.html",
        {
            "user": user,
            "rows": rows,
            "chart": _chart(rows, metric, shooters),
            "guns": trend.guns(session),
            "distances": trend.distances(session),
            "shooters": shooters,
            "metrics": {key: title for key, (title, _) in METRICS.items()},
            "owner": trend.OWNER,
            # The parsed distance, not the raw one: `?distance=25` filters fine but
            # never equals the option's "25.0", so the select would read "All".
            "f_gun": gun,
            "f_distance": str(distance_m) if distance_m is not None else "",
            "f_shooter": shooter,
            "f_metric": metric,
        },
    )
