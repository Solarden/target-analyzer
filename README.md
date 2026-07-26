# target-analyzer

Self-hosted scoring of photos of paper shooting targets: load a photo, register
it, mark the hits, and track score + group/precision metrics over time. The
deeper goal is a sandbox for image interpretation — comparing classic CV, a VLM,
and a custom YOLO on identical data.

> **Status: early.** In place so far: the shared wire contract, the full database
> schema, and the scoring/metrics core (rings, group size, precision, bias). Not
> yet: the HTTP API, the Mac client, and the dashboard. The build proceeds phase by
> phase. Design and roadmap notes live in the maintainer's private docs (symlinked
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
uv run pytest
```

- Dev database is SQLite (zero setup); production is the shared Postgres on the Pi.
- Config is environment-driven (`TA_*`). Copy `.env.example` to `.env` for local
  overrides — real secrets never live in the repo.

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md). A
commercial license is available; details in LICENSING.md.
