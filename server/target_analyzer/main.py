"""The FastAPI app.

Two surfaces on one process: the machine API (``POST /api/ingest``, bearer token) and
the human dashboard (cookie session, ``require_user`` on every route). The wiring that
holds the second one up lives here — the signing-key guard, the session middleware and
the redirect that turns a missing session into a trip to the login page.
"""

from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from target_analyzer import __version__, api
from target_analyzer.auth import NotAuthenticatedError
from target_analyzer.config import INSECURE_DEFAULT_SECRET, get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    # get_secret_value() is load-bearing: SecretStr never compares equal to a plain
    # string, so the obvious spelling of this check is one that can never fire.
    if settings.secret_key.get_secret_value() == INSECURE_DEFAULT_SECRET and not settings.debug:
        raise RuntimeError(
            "TA_SECRET_KEY is not set (using the insecure default). Set it to a long "
            'random value, e.g. `python -c "import secrets; print(secrets.token_urlsafe(48))"`.'
        )

    # FastAPI registers /docs, /redoc and /openapi.json outside the routers that carry
    # require_user, so they hand any device the full route map before logging in.
    app = FastAPI(
        title="target-analyzer",
        version=__version__,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
    )
    # str(SecretStr) is a mask, not the secret: handing the middleware the wrapper
    # signs every cookie in every install with the same ten asterisks.
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key.get_secret_value(),
        same_site="lax",
        https_only=settings.secure_cookies,
    )
    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).parent / "static")),
        name="static",
    )

    for router in api.routers:
        app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)

    @app.exception_handler(NotAuthenticatedError)
    async def _redirect_to_login(request: Request, exc: NotAuthenticatedError) -> RedirectResponse:
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

    return app


app = create_app()
