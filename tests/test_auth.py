"""The dashboard login (implementation.md §8, §12 item 2).

The property this file exists for is that the boundary is closed by default: every
dashboard surface ends at the login page without a session, and a failed login says
nothing about which half of the credentials was wrong.
"""

from types import SimpleNamespace

import itsdangerous
import pytest
from fastapi import status
from pydantic import SecretStr
from sqlmodel import Session
from starlette.middleware.sessions import SessionMiddleware
from tests.conftest import PASSWORD, USERNAME

from target_analyzer import api
from target_analyzer import auth as auth_module
from target_analyzer.auth import hash_password, login_session, verify_password
from target_analyzer.config import INSECURE_DEFAULT_SECRET, Settings, get_settings
from target_analyzer.main import create_app
from target_analyzer.models import User
from target_analyzer.queries import users

# Read from the app, never hand-kept: a hand-kept list is one someone forgets to extend,
# and the route they forget is the one that ships without a login gate. `/` is in it
# because it redirects into the dashboard, so an open front door shows up here too.
GUARDED = sorted(
    {("GET", "/")}
    | {
        (method, route.path.replace("{session_id}", "1").replace("{image_id}", "1"))
        for router in api.routers
        for route in router.routes
        if route.path.startswith("/dashboard")
        for method in route.methods
    }
)
# An empty parametrize skips silently; this fails collection instead.
assert GUARDED


@pytest.fixture
def dormant(db_session: Session) -> User:
    """A user who was deactivated rather than deleted."""
    row = users.create(
        db_session, username="dormant", name="Dormant", password_hash=hash_password(PASSWORD)
    )
    row.is_active = False
    db_session.add(row)
    db_session.commit()

    return row


def test_a_hashed_password_verifies_and_a_wrong_one_does_not():
    digest = hash_password(PASSWORD)

    assert verify_password(PASSWORD, digest)
    assert not verify_password("something else", digest)


@pytest.mark.parametrize(("method", "path"), GUARDED)
def test_an_unauthenticated_request_ends_at_the_login_page(client, method, path):
    assert client.request(method, path).url.path == "/login"


def test_a_deactivated_user_never_sees_the_logged_in_header(auth_client, user, db_session):
    """The header follows the resolved user, not the cookie: a stale `user_id` would
    otherwise render a Log out button on the login page they were just bounced to.
    """
    user.is_active = False
    db_session.add(user)
    db_session.commit()
    page = auth_client.get("/dashboard")

    assert page.url.path == "/login"
    assert "Log out" not in page.text


@pytest.mark.parametrize(
    ("username", "password"),
    [
        (USERNAME, "not-the-password"),  # real user, wrong password
        ("ghost", PASSWORD),  # no such user
        ("dormant", PASSWORD),  # right password, deactivated account
    ],
)
def test_a_failed_login_never_says_which_half_was_wrong(client, user, dormant, username, password):
    response = client.post("/login", data={"username": username, "password": password})

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid username or password." in response.text


@pytest.mark.parametrize(
    "row",
    [
        None,
        User(id=1, name="D", username="d", password_hash="stored", is_active=False),
        User(id=2, name="L", username="l", password_hash="stored", is_active=True),
    ],
    ids=["no such user", "deactivated", "wrong password"],
)
def test_every_failed_login_pays_the_same_hash(monkeypatch, row):
    """The identical message is only half of the no-leak property: bcrypt is the only
    slow thing on the request, so a login that fails without it answers sooner and says
    what the message will not.
    """
    compared: list[str] = []
    monkeypatch.setattr(
        auth_module, "verify_password", lambda _plain, digest: compared.append(digest) or False
    )

    assert not auth_module.authenticate(row, "x")
    assert len(compared) == 1


def test_the_stand_in_hash_costs_what_a_real_one_costs():
    """Equal work, not merely an equal number of comparisons: a stand-in built at a
    lower cost factor would be the fast path again, in a less visible form.
    """

    def cost(digest: str) -> str:
        return digest.split("$")[2]

    assert cost(auth_module._absent_user_hash()) == cost(hash_password("whatever"))


def test_logging_in_opens_the_dashboard(auth_client):
    assert auth_client.get("/dashboard").status_code == status.HTTP_200_OK


def test_logging_out_closes_it(auth_client):
    auth_client.post("/logout")

    assert auth_client.get("/dashboard").url.path == "/login"


def test_deactivating_a_user_ends_the_session_they_already_hold(auth_client, user, db_session):
    user.is_active = False
    db_session.add(user)
    db_session.commit()

    assert auth_client.get("/dashboard").url.path == "/login"


def test_logging_in_replaces_whatever_was_in_the_session():
    """Session fixation: authenticating must not adopt a session an attacker planted."""
    request = SimpleNamespace(session={"planted": "value"})
    login_session(request, User(id=7, name="T", username="t", password_hash="x"))

    assert request.session == {"user_id": 7}


def test_the_session_cookie_is_signed_with_the_configured_secret(auth_client):
    """``str(SecretStr)`` is a mask, so a wrapper handed to the session middleware would
    sign every cookie in every install with the same ten asterisks — and nothing about
    the app would look broken.
    """
    signer = itsdangerous.TimestampSigner(get_settings().secret_key.get_secret_value())

    assert signer.unsign(auth_client.cookies["session"], max_age=60)


@pytest.mark.parametrize("secure", [True, False])
def test_the_secure_cookie_setting_reaches_the_session_middleware(monkeypatch, secure):
    """The setting is only worth a default if it arrives where the flag is actually set."""
    monkeypatch.setattr(get_settings(), "secure_cookies", secure)
    session = next(m for m in create_app().user_middleware if m.cls is SessionMiddleware)

    assert session.kwargs["https_only"] is secure


def test_the_secure_cookie_flag_defaults_on(monkeypatch):
    """The conftest turns this off, so no other test can notice the default being flipped back."""
    monkeypatch.delenv("TA_SECURE_COOKIES", raising=False)

    assert Settings(_env_file=None).secure_cookies is True


@pytest.mark.parametrize(("debug", "refused"), [(False, True), (True, False)])
def test_the_insecure_default_secret_is_refused_outside_debug(monkeypatch, debug, refused):
    settings = get_settings()
    monkeypatch.setattr(settings, "secret_key", SecretStr(INSECURE_DEFAULT_SECRET))
    monkeypatch.setattr(settings, "debug", debug)

    if not refused:
        assert create_app() is not None

        return

    with pytest.raises(RuntimeError, match="TA_SECRET_KEY"):
        create_app()
