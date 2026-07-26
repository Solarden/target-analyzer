#!/usr/bin/env bash
# Fail if the SQLModel models have drifted from the Alembic migrations, i.e. if
# `alembic revision --autogenerate` would produce a non-empty migration.
#
# Runs against a throwaway temp database (upgraded to head first) so it never
# touches a dev database.
set -euo pipefail

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
export TA_DATABASE_URL="sqlite:///$tmp/alembic_check.db"

uv run alembic upgrade head >/dev/null
uv run alembic check
