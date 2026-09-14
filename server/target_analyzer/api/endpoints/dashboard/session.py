"""One shooting session: the target with its holes drawn on it, and editable notes."""

from typing import Annotated

from fastapi import Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from target_analyzer.api.deps import CurrentUser, DbSession
from target_analyzer.api.endpoints.dashboard import dashboard_router
from target_analyzer.api.endpoints.dashboard.render import geometry
from target_analyzer.models import ShootingSession
from target_analyzer.queries import sessions
from target_analyzer.templating import templates

router = dashboard_router()


class NotesForm(BaseModel):
    # Bounded to what the wire contract already accepts for the same field, so a
    # session cannot be edited into a state the client could never have shipped.
    notes: str = Field(default="", max_length=10_000)


@router.get("/session/{session_id}", response_class=HTMLResponse)
def session_view(
    request: Request, session: DbSession, user: CurrentUser, session_id: int, photo: str = "on"
) -> HTMLResponse:
    found = sessions.for_session(session, session_id)

    if found is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such session")

    shooting_session, overlays = found

    return templates.TemplateResponse(
        request,
        "dashboard/session.html",
        {
            "user": user,
            "session": shooting_session,
            "overlays": [(overlay, geometry(overlay.profile)) for overlay in overlays],
            "show_photo": photo != "off",
        },
    )


@router.post("/session/{session_id}/notes")
def save_notes(
    session: DbSession, session_id: int, form: Annotated[NotesForm, Form()]
) -> RedirectResponse:
    # Nothing is rendered here, so the row is all this needs: building the Session
    # view's overlays would be four queries for a column it never reads.
    shooting_session = session.get(ShootingSession, session_id)

    if shooting_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such session")

    shooting_session.notes = form.notes
    session.add(shooting_session)
    session.commit()

    # Back to the notes box rather than the top of the page: the target above it is a
    # full-width render, so landing at the top hides what was just saved.
    return RedirectResponse(
        f"/dashboard/session/{session_id}#notes", status_code=status.HTTP_303_SEE_OTHER
    )
