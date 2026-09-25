"""The migrations, run for real on Postgres: the one database that can tell a plain
timestamp from a timezone-aware one, and so the only one these tests mean anything on."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

REPO = Path(__file__).resolve().parents[1]
BEFORE_SHOOTER = "9728ec8d1ef0"
# Hand-picked so a wrong conversion is visible: Warsaw is two hours ahead in September.
STORED = datetime(2026, 9, 12, 18, 30, 5)


@pytest.fixture
def migrating(_database: Engine) -> Iterator[tuple[Connection, Config]]:
    """A connection inside a schema of its own, far from the one the suite builds."""
    if _database.dialect.name != "postgresql":
        pytest.skip("SQLite stores every timestamp alike, so there is nothing to convert")

    # An engine of its own, unpooled: the search path and zone set below would otherwise
    # ride a pooled connection back into the rest of the suite.
    engine = create_engine(_database.url, poolclass=NullPool)

    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS migration_check CASCADE"))
        conn.execute(text("CREATE SCHEMA migration_check"))
        conn.execute(text("SET search_path TO migration_check"))
        # Not UTC on purpose: the conversion must not depend on the server's zone.
        conn.execute(text("SET TIME ZONE 'Europe/Warsaw'"))
        conn.commit()
        config = Config(str(REPO / "alembic.ini"))
        config.set_main_option("script_location", str(REPO / "alembic"))
        config.attributes["connection"] = conn

        try:
            yield conn, config
        finally:
            conn.rollback()
            conn.execute(text("DROP SCHEMA migration_check CASCADE"))
            conn.commit()


def test_every_stored_instant_survives_the_timestamptz_migration(migrating):
    conn, config = migrating
    command.upgrade(config, BEFORE_SHOOTER)
    conn.execute(
        text(
            'INSERT INTO "user" (name, username, password_hash, is_active, created_at) '
            "VALUES ('a', 'a', 'x', true, :at)"
        ),
        {"at": STORED},
    )
    conn.commit()

    command.upgrade(config, "head")
    conn.commit()
    read = conn.execute(text('SELECT created_at FROM "user"')).scalar_one()

    assert read == STORED.replace(tzinfo=UTC)


def test_the_migration_adds_an_empty_shooter_and_undoes_cleanly(migrating):
    conn, config = migrating
    command.upgrade(config, "head")
    conn.commit()
    columns = dict(
        conn.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'migration_check' AND table_name = 'session'"
            )
        ).all()
    )

    assert columns["shooter"] == "character varying"
    assert columns["created_at"] == "timestamp with time zone"

    command.downgrade(config, BEFORE_SHOOTER)
    conn.commit()
    after = conn.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'migration_check' AND table_name = 'session'"
        )
    ).scalars()

    assert "shooter" not in set(after)
