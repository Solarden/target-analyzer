#!/usr/bin/env bash
# Fail if the SQLModel models have drifted from the Alembic migrations, i.e. if
# `alembic revision --autogenerate` would produce a non-empty migration.
#
# Runs against a throwaway temp SQLite database, upgraded to head first. SQLite stores every
# datetime alike, so a column-type drift needs ALEMBIC_CHECK_DATABASE_URL: a disposable Postgres.
#
# Not TA_DATABASE_URL, which may already name a real database in the calling shell.
set -euo pipefail

if [ -n "${ALEMBIC_CHECK_DATABASE_URL:-}" ]; then
  export TA_DATABASE_URL="$ALEMBIC_CHECK_DATABASE_URL"
else
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  export TA_DATABASE_URL="sqlite:///$tmp/alembic_check.db"
fi

uv run alembic upgrade head >/dev/null
uv run alembic check
