from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlmodel import Session

from target_analyzer.db import get_engine

router = APIRouter(tags=["meta"])


# Also the container healthcheck, which only marks the container unhealthy rather than
# restarting it. Kept out of the docstring: that text is published in the OpenAPI schema.
@router.get("/health")
def health() -> JSONResponse:
    """Report whether the service is running and its database is reachable.

    Returns 503 when the database cannot be reached: the service cannot store anything
    without it, so it is not healthy.
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
