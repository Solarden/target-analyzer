"""Every reading of one photo, side by side (implementation.md §10).

The template loops over the readings, and each one that is not the hand-marked ground
truth is also measured against it: matched, missed and spurious holes, which is what
says whether a detector is improving rather than only what it drew.
"""

from fastapi import HTTPException, Request, status
from fastapi.responses import HTMLResponse

from ta_shared.agreement import MATCH_TOL_MM, agreement
from ta_shared.profile import mm_per_px
from target_analyzer.api.deps import CurrentUser, DbSession
from target_analyzer.api.endpoints.dashboard import dashboard_router
from target_analyzer.api.endpoints.dashboard.render import geometry
from target_analyzer.queries import sessions
from target_analyzer.templating import templates

router = dashboard_router()


def _points(overlay: sessions.Overlay) -> list[tuple[float, float]]:
    return [(hole.x_canon, hole.y_canon) for hole in overlay.holes]


def _versus_truth(overlays: list[sessions.Overlay]) -> dict[int, dict[str, object]]:
    """Every other reading measured against the hand-marked one, by interpretation id."""
    truth = next(
        (o for o in overlays if o.interpretation.method == sessions.GROUND_TRUTH_METHOD), None
    )
    scale = mm_per_px(truth.profile) if truth is not None else None

    # No hand-marked reading, or no physical scale to state the tolerance in: there is
    # nothing to measure against, which is different from measuring zero agreement.
    if truth is None or scale is None:
        return {}

    truth_points = _points(truth)
    measured = {}

    for overlay in overlays:
        # Different profile versions are different geometry, so these coordinates are not
        # comparable and any millimetre figure over them would be invented.
        if overlay is truth or overlay.profile.id != truth.profile.id:
            continue

        result = agreement(truth_points, _points(overlay), MATCH_TOL_MM / scale)
        measured[overlay.interpretation.id] = {
            "matched": result.matched,
            "missed": result.missed,
            "spurious": result.spurious,
            "offset_mm": (None if result.mean_offset_px is None else result.mean_offset_px * scale),
        }

    return measured


@router.get("/compare/{image_id}", response_class=HTMLResponse)
def compare_view(
    request: Request, session: DbSession, user: CurrentUser, image_id: int, photo: str = "on"
) -> HTMLResponse:
    found = sessions.for_image(session, image_id)

    if found is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such image")

    image, shooting_session, overlays = found

    return templates.TemplateResponse(
        request,
        "dashboard/compare.html",
        {
            "user": user,
            "image": image,
            "session": shooting_session,
            "overlays": [(overlay, geometry(overlay.profile)) for overlay in overlays],
            "versus": _versus_truth(overlays),
            "show_photo": photo != "off",
        },
    )
