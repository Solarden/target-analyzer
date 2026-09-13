"""Engine + session factory. Copied from expense-analyzer and kept dialect-aware.

Dev runs on SQLite (zero setup), the Pi runs on the shared Postgres — one model set
serves both because every column type in models.py is generic (see §5).
"""

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine, make_url
from sqlmodel import Session, create_engine

from target_analyzer.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """Build (once) and return the database engine.

    Lazy on purpose: importing this module must not open a connection, so the
    database URL can still be overridden (e.g. by tests) before first use.
    """
    settings = get_settings()
    url = make_url(settings.database_url)

    if url.get_backend_name() == "sqlite":
        if url.database and url.database != ":memory:":
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)

        engine = create_engine(
            settings.database_url,
            echo=settings.debug,
            # SQLite + threaded server: allow connections across threads.
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record):
            # WAL mode for write safety and better concurrency. foreign_keys is off by
            # default in SQLite and must be enabled per-connection — without it the
            # ON DELETE CASCADE on image/interpretation/hole is silently inert.
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

        return engine

    # Server database (PostgreSQL). Small fixed pool. pre_ping matters: the database
    # lives in a separate compose stack and can restart independently of the app, so
    # stale pooled connections must be detected, not crashed on.
    return create_engine(
        settings.database_url,
        echo=settings.debug,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_recycle=1800,
    )


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a database session."""
    with Session(get_engine()) as session:
        yield session
