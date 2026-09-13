"""The FastAPI app.

Deliberately bare: P2 serves the machine API only. Cookie sessions
(``SessionMiddleware``), the human login and the fail-closed ``TA_SECRET_KEY`` guard
arrive together in P5, with the dashboard that needs them — a signing key that signs
nothing is not yet a boundary worth guarding.
"""

from fastapi import FastAPI

from target_analyzer import __version__, api


def create_app() -> FastAPI:
    app = FastAPI(title="target-analyzer", version=__version__)

    for router in api.routers:
        app.include_router(router)

    return app


app = create_app()
