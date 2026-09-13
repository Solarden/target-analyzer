# target-analyzer

Self-hosted scoring of photos of paper shooting targets: load a photo, register
it, mark the hits, and track score + group/precision metrics over time. The
deeper goal is a sandbox for image interpretation — comparing classic CV, a VLM,
and a custom YOLO on identical data.

> **Status: early.** In place so far: the shared wire contract, the full database
> schema, the scoring/metrics core (rings, group size, precision, bias), and the
> ingest API — `POST /api/ingest` behind a machine bearer token, idempotent on the
> image's sha256. Not yet: the Mac client and the dashboard. The build proceeds phase
> by phase. Design and roadmap notes live in the maintainer's private docs (symlinked
> locally at `internal_docs/`, not part of this public repo).

## Architecture

Two pieces:

- **Mac — the processing client.** Loads a photo, registers it to a fixed
  *canonical frame* via homography (an ArUco marker board, least-squares fit),
  marks hits by clicking, and ships canonical `(x, y)` coordinates to the server.
  An offline outbox queues submissions when the server is unreachable.
- **Raspberry Pi — the server.** A FastAPI app that turns coordinates
  into ring scores + metrics from the (versioned) target profile, persists to
  Postgres, and serves an HTMX dashboard behind a login.

The Mac never scores — it sends coordinates and the server owns the scoring
geometry, so re-measuring a target re-scores every past session consistently.

## Layout

```
shared/ta_shared/         # the wire contract (payload + profile) — imported by both sides
client/ta_client/         # the Mac processing client
server/target_analyzer/   # the Pi FastAPI app
profiles/                 # versioned target profiles (ring geometry + ArUco board)
alembic/                  # database migrations
tests/
```

## Development

```
uv sync --extra dev --extra server --extra client
uv run pre-commit install
docker compose -f docker-compose.test.yml up -d --wait   # throwaway Postgres for the tests
uv run pytest
```

- Dev database is SQLite (zero setup); production is the shared Postgres on the Pi.
  The DB-backed tests run against a throwaway Postgres for dialect parity — set
  `TA_TEST_DATABASE_URL` to a SQLite URL for a docker-less run.
- Config is environment-driven (`TA_*`). Copy `.env.example` to `.env` for local
  overrides — real secrets never live in the repo.

### Running the server

```
uv run alembic upgrade head
uv run python -m target_analyzer.seed_profile profiles/issf_precision.json
uv run python -m target_analyzer.create_token   # token -> the Mac, hash -> TA_INGEST_TOKEN_HASH
uv run uvicorn target_analyzer.main:app --reload
```

A target profile must be seeded before anything can be ingested: it is the geometry
the server scores against, and profiles are immutable and versioned, so re-measuring
the physical target bumps the version rather than editing the old row.

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md). A
commercial license is available; details in LICENSING.md.
