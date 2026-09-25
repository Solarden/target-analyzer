# target-analyzer

Self-hosted scoring of photos of paper shooting targets: load a photo, register
it, mark the hits, and track score + group/precision metrics over time. The
deeper goal is a sandbox for image interpretation — comparing classic CV, a VLM,
and a custom YOLO on identical data.

> **Status: early.** In place so far: the shared wire contract, the full database
> schema, the scoring/metrics core (rings, group size, precision, bias), the ingest
> API — `POST /api/ingest` behind a machine bearer token, idempotent on the image's
> sha256 — the Mac client end to end, from photo to a shipped session, with an offline
> outbox behind it, and the dashboard behind a login: score and precision over time,
> the target rendered with its holes, and one photo's readings side by side, and a
> container image with a compose stack that serves it behind TLS. Not yet: anything that
> reads the holes for you. The build proceeds phase by phase.

## Architecture

Two pieces:

- **Mac — the processing client.** Loads a photo, registers it to a fixed
  *canonical frame* via homography (an ArUco marker board, least-squares fit),
  marks hits by clicking, and ships canonical `(x, y)` coordinates to the server.
  An offline outbox queues submissions when the server is unreachable.
- **Raspberry Pi — the server.** A FastAPI app that turns coordinates
  into ring scores + metrics from the (versioned) target profile, persists to
  Postgres, and serves a server-rendered dashboard behind a login.

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

- `uv sync --extra yolo` is separate and optional: it pulls torch to train and run the
  custom detector, and nothing in the manual or classic-CV path needs it.
- Dev database is SQLite (zero setup); production is Postgres.
  The DB-backed tests run against a throwaway Postgres for dialect parity — set
  `TA_TEST_DATABASE_URL` to a SQLite URL for a docker-less run.
- Config is environment-driven (`TA_*`). Copy `.env.example` to `.env` for local
  overrides — real secrets never live in the repo. Plain-http dev also wants
  `TA_SECURE_COOKIES=false`, or the browser drops the session cookie and every page
  bounces back to the login.

### Running the server

```
uv run alembic upgrade head
uv run python -m target_analyzer.seed_profile profiles/issf_precision.json
uv run python -m target_analyzer.create_token   # token -> the Mac, hash -> TA_INGEST_TOKEN_HASH
uv run python -m target_analyzer.create_user --username alice --name "Alice"
uv run uvicorn target_analyzer.main:app --reload
uv run python -m target_analyzer.export_corpus corpus/sessions   # every marked reading, as session folders
```

Over plain http, set `TA_SECURE_COOKIES=false` first. The session cookie is marked Secure
by default, and the browser drops a Secure cookie on an http origin — so a correct login
lands straight back on the login page with nothing to say why.

A target profile must be seeded before anything can be ingested: it is the geometry
the server scores against, and profiles are immutable and versioned, so re-measuring
the physical target bumps the version rather than editing the old row.

The dashboard is at `/dashboard`, behind the login. There is no public registration —
`create_user` is how an account is made. Set `TA_SECRET_KEY` to a long random value
before starting: the app refuses to sign session cookies with the placeholder unless
`TA_DEBUG=true`.

### The dashboard's CSS

`static/app.css` is generated from `tailwind.css` and **committed**, so running the app
needs no build step. After changing a template, rebuild it:

```
make css
```

The Tailwind standalone CLI is downloaded into `tools/` on demand (gitignored); no Node
is involved. Chart.js is vendored in `static/` for the same reason — nothing on this
dashboard is fetched from a CDN.

### Running the client

```
uv run python -m ta_client.board profiles/issf_precision.json --out board.svg
uv run python -m ta_client.selfcheck
uv run python -m ta_client photo.jpg --gun "CZ 75" --distance 25 --profile profiles/issf_precision.json
uv run python -m ta_client.ship          # drain the outbox without processing a photo
uv run python -m ta_client.cv_blob out/<session>   # score the blob detector against a marked session
uv run python -m ta_client.vlm out/<session>       # same, for the vision model
uv run python -m ta_client.cv_blob_vlm out/<session>  # same, for the two together
uv run python -m ta_client.yolo out/<session>      # same, for the trained model
uv run python -m ta_client.yolo_data corpus/sessions/* --out corpus/yolo --profile profiles/issf_pistol_50m.json
```

The client reads its settings from `~/.config/target-analyzer/env` — never from the
checkout, which is public:

```
TA_SERVER_URL=https://target.example.com
TA_INGEST_TOKEN=the-token-create_token-printed
TA_SHIP_ORIGINAL=true

# For --method cv_blob_vlm and the two vision-model scoring tools above: any
# OpenAI-compatible endpoint that accepts an image.
TA_VLM_BASE_URL=https://vision.example.com/v1
TA_VLM_MODEL=the-model-tag
TA_VLM_API_KEY=only-if-the-endpoint-wants-one

# Only for the trained detector, and only with `uv sync --extra yolo` installed.
TA_YOLO_WEIGHTS=/path/to/the-model.pt
TA_YOLO_DEVICE=          # blank lets the runtime choose
TA_YOLO_CONF=0.10
```

`chmod 600` that file: it holds the bearer token. With `TA_SERVER_URL` unset the client
still processes photos and leaves them queued.

Print `board.svg` at 100% — it is in millimetres, and the calibration line on it must
measure 50 mm with a ruler, or the rings on paper will not match the profile the server
scores against. Photograph the target from as close as you can while keeping the whole
sheet in frame — walk up to it rather than shooting from the firing line. The warp makes
every photo the same 1500 px square, but it cannot invent detail the camera never caught,
and a hole is a few millimetres across. Then run the pipeline: it warps the photo into
the canonical frame, asks you to confirm the rings landed on the printed ones, and lets
you click the holes. The session then goes to the
server; if the server is unreachable it stays in the outbox
(`~/.target-analyzer/outbox/`, overridable with `--outbox`) and the next run sends it,
oldest first. Ingest is idempotent on the photo's sha256, so a replayed flush never
creates a second session.

`--method cv_blob` runs the blob detector first and opens the hole picker on what it
found, to drag, delete or add to.

`--method cv_blob_vlm` adds a vision model, because a picture of a target is mostly things
that are hole-shaped and are not holes: the blob filter proposes, the model is shown a close
crop of each proposal and asked whether it is really a hole, and the survivors keep the blob
filter's coordinates. The model is not offered on its own — shown a whole frame, it cannot
place a hole precisely enough to score it. It costs one model call per proposal rather than
one per photo, so it is the slowest method by some way.

`--method yolo` runs a model trained on this project's own photos. It needs
`uv sync --extra yolo` and `TA_YOLO_WEIGHTS` pointing at a trained model — there is none in
the repo, and `ta_client.yolo_data` is what builds a dataset from marked sessions and
prints the command that trains one. It reads the frame in 100 mm tiles rather than whole:
a hole is thirteen pixels in a 1500 px square, and a detector that resizes the square to
its own input sees five.

Both readings are then sent — the detector's untouched one and yours — and the dashboard's
Compare view puts them side by side with the difference between them counted. Expect to
correct any of them: a hole and last week's patched hole look alike, so on a reused target
much of what a detector proposes is not from this string.

`--profile` has no default on purpose: it is the geometry every hit is scored against,
and the wrong one produces a complete, plausible session in which every shot is in the
wrong ring.

macOS only: HEIC photos are converted with `sips` on the way in.

## Deployment

`docker compose up -d` builds the image, applies the migrations on start and serves the
dashboard over HTTPS. It does **not** run a database: point it at an existing PostgreSQL
server, and create the role and the database there first, as its superuser.

```
CREATE ROLE target_analyzer LOGIN PASSWORD '...';
CREATE DATABASE target_analyzer OWNER target_analyzer;
```

Copy `.env.example` to `.env` beside the compose file and fill in `TA_DATABASE_URL`,
`TA_SECRET_KEY` and `TA_SITE_ADDRESS`. A host that already runs a reverse proxy starts
the app on its own instead, and points that proxy at port 8000:

```
docker compose up -d app
```

Then two one-time steps — the geometry the server scores against (the same profile you
print the sheet from and pass to the client) and an account to log in with:

```
docker compose run --rm app python -m target_analyzer.seed_profile profiles/issf_pistol_50m.json
docker compose run --rm app python -m target_analyzer.create_user --username alice --name "Alice"
```

`create_user` prompts for the password twice, so it needs a terminal — `docker compose
run` gives it one, `exec` on the running container does not. Last, mint the client's
bearer token: the hash goes in `.env`, the token itself in the Mac's
`~/.config/target-analyzer/env`.

```
docker compose run --rm app python -m target_analyzer.create_token
```

Upgrading is `docker compose build app && docker compose up -d app`: the container runs
`alembic upgrade head` before uvicorn, so migrations need no step of their own, and
`/health` answers 200 only once they have applied and the database is reachable — which
is also what the container's healthcheck asks.

`tls internal` means Caddy signs its own certificate; browsers trust it once its root CA
is installed on the devices you browse from. For a publicly trusted certificate without
exposing the host, use an ACME DNS-01 challenge instead — that needs a Caddy image built
with your DNS provider's plugin.

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md). A
commercial license is available; details in LICENSING.md.
