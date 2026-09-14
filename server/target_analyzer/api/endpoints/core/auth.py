"""Login and logout. The only routes a logged-out browser may reach."""

from typing import Annotated

from fastapi import APIRouter, Form, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, SecretStr

from target_analyzer.api.deps import DbSession
from target_analyzer.auth import authenticate, login_session, logout_session
from target_analyzer.queries import users
from target_analyzer.templating import templates

# No prefix and no router-level dependency: this is the one surface that has to answer
# before there is a session to check.
router = APIRouter(tags=["auth"])


class LoginForm(BaseModel):
    username: str
    password: SecretStr  # masked in repr and tracebacks; read via .get_secret_value()


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "core/login.html", {})


@router.post("/login")
def login(request: Request, form: Annotated[LoginForm, Form()], session: DbSession) -> Response:
    user = users.by_username(session, form.username.strip())

    # One message for all three failures, and authenticate() makes them cost the same
    # too — an unknown username that answers faster says what the message will not.
    if not authenticate(user, form.password.get_secret_value()):
        return templates.TemplateResponse(
            request,
            "core/login.html",
            {"error": "Invalid username or password."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    login_session(request, user)

    return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/logout")
def logout(request: Request) -> RedirectResponse:
    logout_session(request)

    return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
