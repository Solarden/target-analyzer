"""The FastAPI app.

Two surfaces on one process: the machine API (``POST /api/ingest``, bearer token) and
the human dashboard (cookie session, ``require_user`` on every route). The wiring that
holds the second one up lives here — the signing-key guard, the session middleware and
the redirect that turns a missing session into a trip to the login page.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response

from target_analyzer import __version__, api
from target_analyzer.auth import NotAuthenticatedError
from target_analyzer.config import INSECURE_DEFAULT_SECRET, get_settings

# Set here rather than in the shipped Caddyfile because not every deployment is served
# through that file — one behind its own proxy would otherwise get none of this.
SECURITY_HEADERS = {
    # 'unsafe-inline' is for the two onchange="this.form.submit()" handlers on the Trend
    # filters; no template carries a style attribute, so style-src needs no exception.
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self'; "
        "img-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    ),
    # Same-origin forms throughout, so framing is never legitimate — and SameSite=Lax does
    # not help inside a frame, where a POST is same-site.
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    # Browsers ignore this over plain http, so it costs local dev nothing.
    "Strict-Transport-Security": "max-age=31536000",
}


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

    # Registered after the session middleware, so it wraps it and covers error responses
    # and redirects too, not just the routes below.
    @app.middleware("http")
    async def _add_security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)

        return response

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
