"""Shared test fixtures.

The DB-backed tests run against a real PostgreSQL by default (prod parity): the
throwaway container from ``docker-compose.test.yml``. Set ``TA_TEST_DATABASE_URL``
(e.g. a sqlite URL) for a quick docker-less run.

The database fixture is deliberately **not autouse**: the scoring and contract tests are
pure and must keep running with no container up. Only the tests that ask for
``db_session`` / ``client`` pay for one.

Redirecting the URL here is safe because the engine is built lazily (see
``target_analyzer.db.get_engine``) — nothing opens a connection at import time.
"""

import hashlib
import os
import tempfile
from collections.abc import Iterator
from datetime import date
from io import BytesIO
from pathlib import Path

import bcrypt
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, SQLModel

from ta_shared.payload import Hit, SessionMeta, ShipPayload
from target_analyzer.auth import hash_password
from target_analyzer.config import get_settings
from target_analyzer.models import TargetProfile, User
from target_analyzer.queries import users
from target_analyzer.seed_profile import seed

# The token the suite authenticates with; only its hash is ever configured, exactly as
# in production. Not a credential — it never leaves this file.
INGEST_TOKEN = "test-machine-token"  # nosec B105
# The dashboard login, on the same terms.
USERNAME = "tester"
PASSWORD = "secret123"  # nosec B105

_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="ta-test-"))
os.environ["TA_DATABASE_URL"] = os.environ.get(
    "TA_TEST_DATABASE_URL",
    "postgresql+psycopg://ta_test:ta_test@localhost:55433/ta_test",
)
# Uploaded images land in a throwaway temp dir, never the real data/.
os.environ["TA_DATA_PATH"] = str(_TEST_DATA_DIR)
os.environ["TA_INGEST_TOKEN_HASH"] = hashlib.sha256(INGEST_TOKEN.encode()).hexdigest()
os.environ["TA_SECRET_KEY"] = "test-secret-not-for-production"  # nosec B105
# Not optional: over plain http the TestClient drops a Secure cookie, so a developer's
# local value of true would 303 every authenticated test to /login with no clue why.
os.environ["TA_SECURE_COOKIES"] = "false"
# os.environ outranks the dotenv file in pydantic-settings, so a developer's local one
# cannot leak into the suite through the vars set above.
get_settings.cache_clear()

PROFILE_JSON = Path(__file__).parent.parent / "profiles" / "issf_precision.json"


@pytest.fixture(scope="session")
def _database() -> Iterator[Engine]:
    """Connectivity gate + one schema for the whole run."""
    from target_analyzer.db import get_engine

    engine = get_engine()

    try:
        with engine.connect():
            pass
    except OperationalError as exc:
        pytest.exit(
            f"test database unreachable ({engine.url.render_as_string()}) — run "
            "`docker compose -f docker-compose.test.yml up -d --wait`, "
            f"or set TA_TEST_DATABASE_URL: {exc}",
            returncode=4,
        )

    SQLModel.metadata.drop_all(engine)  # leftovers from an aborted earlier run
    SQLModel.metadata.create_all(engine)

    try:
        yield engine
    finally:
        SQLModel.metadata.drop_all(engine)

        # scripts/alembic_check.sh may share this database: a surviving version row makes its
        # next `upgrade head` a no-op over the tables just dropped.
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version"))


def _reset_all_tables(engine: Engine) -> None:
    """Wipe every table (and restart id sequences) between tests."""
    tables = SQLModel.metadata.sorted_tables

    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            names = ", ".join(f'"{t.name}"' for t in tables)
            conn.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
        else:
            conn.execute(text("PRAGMA defer_foreign_keys=ON"))

            for table in reversed(tables):
                conn.execute(table.delete())


@pytest.fixture
def db_session(_database: Engine) -> Iterator[Session]:
    try:
        with Session(_database) as session:
            yield session
    finally:
        _reset_all_tables(_database)


@pytest.fixture(scope="session", autouse=True)
def _fast_bcrypt() -> Iterator[None]:
    """Hash at the cheapest cost factor for the suite.

    Every login here runs through the real route, and real cost-12 bcrypt makes that
    the slowest thing in the run.
    """
    original = bcrypt.gensalt

    bcrypt.gensalt = lambda rounds=4, prefix=b"2b": original(4, prefix)

    try:
        yield
    finally:
        bcrypt.gensalt = original


@pytest.fixture
def client(db_session: Session) -> Iterator[TestClient]:
    # Imported here rather than at module scope: create_app() runs the fail-closed
    # secret check, and the settings it reads are written above this fixture.
    from target_analyzer.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def user(db_session: Session) -> User:
    return users.create(
        db_session, username=USERNAME, name="Tester", password_hash=hash_password(PASSWORD)
    )


@pytest.fixture
def auth_client(client: TestClient, user: User) -> TestClient:
    """A client that logged in the way a browser does, so the session cookie is real."""
    response = client.post(
        "/login", data={"username": USERNAME, "password": PASSWORD}, follow_redirects=False
    )

    assert response.status_code == status.HTTP_303_SEE_OTHER

    return client


@pytest.fixture
def auth() -> dict[str, str]:
    """Headers carrying the valid machine token."""
    return {"Authorization": f"Bearer {INGEST_TOKEN}"}


@pytest.fixture
def profile(db_session: Session) -> TargetProfile:
    """The seeded ISSF profile — the real JSON artifact, through the real seeder."""
    row, _created = seed(db_session, PROFILE_JSON)

    return row


# --- image + payload builders ---------------------------------------------------


def make_png(size: int = 1000, colour: str = "white") -> bytes:
    """A valid PNG of ``size`` x ``size`` — stands in for the normalized render."""
    buf = BytesIO()
    PILImage.new("RGB", (size, size), colour).save(buf, format="PNG")

    return buf.getvalue()


def make_jpeg(size: int = 64, *, with_exif: bool = False) -> bytes:
    """A valid JPEG, optionally carrying EXIF so the server-side strip is provable."""
    buf = BytesIO()
    img = PILImage.new("RGB", (size, size), "grey")

    if with_exif:
        exif = PILImage.Exif()
        exif[0x010F] = "TestCamera"  # Make
        exif[0x0110] = "Model X"  # Model
        img.save(buf, format="JPEG", exif=exif)
    else:
        img.save(buf, format="JPEG")

    return buf.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    return make_png()


@pytest.fixture
def jpeg_bytes() -> bytes:
    return make_jpeg()


def make_payload(image_bytes: bytes, **overrides) -> ShipPayload:
    """A well-formed payload for ``image_bytes``, with the hash already correct."""
    fields = {
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "canon_size_px": 1000,
        "session": SessionMeta(
            gun="CZ 75",
            distance_m=25,
            shot_at=date(2026, 7, 26),
            target_profile="issf_precision",
            target_profile_version=1,
        ),
        # A tight group down-and-right of centre — the doc's worked example.
        "hits": [Hit(x_canon=500, y_canon=500), Hit(x_canon=510, y_canon=500)],
    }

    return ShipPayload(**(fields | overrides))


def post_ingest(client, auth, payload, png, *, jpeg=None):
    """POST one session the way the Mac client does."""
    files = {"normalized": ("n.png", png, "image/png")}

    if jpeg is not None:
        files["original"] = ("o.jpg", jpeg, "image/jpeg")

    return client.post(
        "/api/ingest", headers=auth, data={"payload": payload.model_dump_json()}, files=files
    )


@pytest.fixture
def ingested(
    client: TestClient, auth: dict[str, str], profile: TargetProfile, png_bytes: bytes
) -> dict:
    """One session in the database, put there through the real ingest path.

    The dashboard tests read what a real flush leaves behind rather than hand-built
    rows, so a change to how scoring is persisted fails them too.
    """
    response = post_ingest(client, auth, make_payload(make_jpeg()), png_bytes)

    assert response.status_code == status.HTTP_201_CREATED

    return response.json()
