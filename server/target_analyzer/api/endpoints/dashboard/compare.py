"""Every reading of one photo, side by side (implementation.md §10).

The MVP ships a single method, so there is one column. The template loops, which is
the whole point: a Phase-2 detector becomes one more interpretation row and appears
here with no change to this view.
"""

from fastapi import HTTPException, Request, status
from fastapi.responses import HTMLResponse

from target_analyzer.api.deps import CurrentUser, DbSession
from target_analyzer.api.endpoints.dashboard import dashboard_router
from target_analyzer.api.endpoints.dashboard.render import geometry
from target_analyzer.queries import sessions
from target_analyzer.templating import templates

router = dashboard_router()


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
            "show_photo": photo != "off",
        },
    )
