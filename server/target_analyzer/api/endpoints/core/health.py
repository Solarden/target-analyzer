from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlmodel import Session

from target_analyzer.db import get_engine

router = APIRouter(tags=["meta"])


@router.get("/health")
def health() -> JSONResponse:
    """Liveness + DB reachability check (also the Docker healthcheck in P6).

    503 when the database is unreachable — it is a server that can go down
    independently of the app, and an ingest endpoint with nowhere to persist is not
    healthy. Docker only *marks* the container unhealthy (no restart loop).
    """
    db_ok = True

    try:
        with Session(get_engine()) as session:
            session.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    return JSONResponse(
        status_code=status.HTTP_200_OK if db_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ok" if db_ok else "degraded",
            "database": "ok" if db_ok else "unreachable",
            "dialect": get_engine().dialect.name,
        },
    )
