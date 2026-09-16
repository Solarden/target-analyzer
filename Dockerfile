# Built with uv. Runs on linux/arm64 as well as amd64 dev machines.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# alembic.ini resolves script_location and prepend_sys_path relative to the working
# directory, so every command that touches migrations has to run from here.
WORKDIR /app

# Dependencies first, WITHOUT the project, so an app-source change below cannot re-run this
# layer — only pyproject.toml / uv.lock bust it.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --no-install-project --no-dev --extra server

ENV PATH="/app/.venv/bin:$PATH"

# App source + migrations last: the only layers a code deploy rebuilds.
COPY shared ./shared
COPY server ./server
# Copied although its extra is never installed — the wheel declares all three packages, and
# hatchling fails the build on a directory it cannot find.
COPY client ./client
COPY alembic.ini ./
COPY alembic ./alembic
# seed_profile reads these at runtime, and they live outside the packaged code.
COPY profiles ./profiles
RUN uv sync --no-dev --extra server

EXPOSE 8000

CMD ["uvicorn", "target_analyzer.main:app", "--host", "0.0.0.0", "--port", "8000"]
