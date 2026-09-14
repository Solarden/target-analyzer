# target-analyzer

Self-hosted scoring of photos of paper shooting targets: load a photo, register
it, mark the hits, and track score + group/precision metrics over time. The
deeper goal is a sandbox for image interpretation — comparing classic CV, a VLM,
and a custom YOLO on identical data.

> **Status: early.** In place so far: the shared wire contract, the full database
> schema, the scoring/metrics core (rings, group size, precision, bias), the ingest
> API — `POST /api/ingest` behind a machine bearer token, idempotent on the image's
> sha256 — and the Mac client end to end, from photo to a shipped session, with an
> offline outbox behind it. Not yet: the dashboard. The build proceeds phase by phase.

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

- Dev database is SQLite (zero setup); production is Postgres.
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

### Running the client

```
uv run python -m ta_client.board profiles/issf_precision.json --out board.svg
uv run python -m ta_client.selfcheck
uv run python -m ta_client photo.jpg --gun "CZ 75" --distance 25 --profile profiles/issf_precision.json
uv run python -m ta_client.ship          # drain the outbox without processing a photo
```

The client reads its settings from `~/.config/target-analyzer/env` — never from the
checkout, which is public:

```
TA_SERVER_URL=https://target.example.com
TA_INGEST_TOKEN=the-token-create_token-printed
TA_SHIP_ORIGINAL=true
```

`chmod 600` that file: it holds the bearer token. With `TA_SERVER_URL` unset the client
still processes photos and leaves them queued.

Print `board.svg` at 100% — it is in millimetres, and the calibration line on it must
measure 50 mm with a ruler, or the rings on paper will not match the profile the server
scores against. Photograph the target so the whole sheet is in frame, then run the
pipeline: it warps the photo into the canonical frame, asks you to confirm the rings
landed on the printed ones, and lets you click the holes. The session then goes to the
server; if the server is unreachable it stays in the outbox
(`~/.target-analyzer/outbox/`, overridable with `--outbox`) and the next run sends it,
oldest first. Ingest is idempotent on the photo's sha256, so a replayed flush never
creates a second session.

`--profile` has no default on purpose: it is the geometry every hit is scored against,
and the wrong one produces a complete, plausible session in which every shot is in the
wrong ring.

macOS only: HEIC photos are converted with `sips` on the way in.

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md). A
commercial license is available; details in LICENSING.md.
