"""Password hashing and the login-session dependency (implementation.md §8).

Single household: any active user, once logged in, sees the same data. There are no
roles, so there is nothing to authorize beyond "is there a valid session" — the
machine API's bearer token (:mod:`target_analyzer.bearer`) is the other, separate
half of the story.

The signed cookie (Starlette ``SessionMiddleware``) holds the user id and nothing
else. ``SameSite=Lax`` on that cookie is the whole CSRF story: it keeps the cookie
off cross-site POSTs, which is what the state-changing routes need.
"""

import secrets
from functools import lru_cache

import bcrypt
from fastapi import Depends, Request
from sqlmodel import Session

from target_analyzer.db import get_session
from target_analyzer.models import User

_SESSION_USER_KEY = "user_id"


class NotAuthenticatedError(Exception):
    """Raised by :func:`require_user` when no valid session is present.

    ``main.py`` handles it by redirecting the browser to the login page.
    """


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, password_hash: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), password_hash.encode("utf-8"))


@lru_cache(maxsize=1)
def _absent_user_hash() -> str:
    """A hash of a value nobody can present, for a login with no account to check against.

    Built through :func:`hash_password` so it carries the cost factor the stored hashes
    do — a cheaper stand-in would be a fast path again, just a less obvious one.
    """
    return hash_password(secrets.token_urlsafe(32))


def authenticate(user: User | None, password: str) -> bool:
    """Whether ``password`` opens ``user``'s account — taking the same time either way.

    An unknown or deactivated account is checked against a throwaway hash instead of
    returning early: bcrypt is the only slow thing on the request, so skipping it
    announces that the account is not there, whatever the message says.
    """
    active = user is not None and user.is_active
    password_hash = user.password_hash if active else _absent_user_hash()
    correct = verify_password(password, password_hash)

    return active and correct


def login_session(request: Request, user: User) -> None:
    # Clear first: authenticating into a session an anonymous visitor already carries
    # would let a planted cookie survive the login as a fixated one.
    request.session.clear()
    request.session[_SESSION_USER_KEY] = user.id


def logout_session(request: Request) -> None:
    request.session.pop(_SESSION_USER_KEY, None)


def current_user(request: Request, session: Session = Depends(get_session)) -> User | None:
    """The logged-in user, or None. An unknown or deactivated one counts as logged out."""
    user_id = request.session.get(_SESSION_USER_KEY)

    if user_id is None:
        return None

    user = session.get(User, user_id)

    # Checked on every request rather than only at login, so deactivating a user
    # ends the sessions they already hold.
    if user is None or not user.is_active:
        return None

    return user


def require_user(user: User | None = Depends(current_user)) -> User:
    """Dependency that enforces login on the dashboard routes."""
    if user is None:
        raise NotAuthenticatedError

    return user
