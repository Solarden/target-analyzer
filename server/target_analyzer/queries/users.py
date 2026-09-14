"""Login-identity lookups."""

from sqlmodel import Session, select

from target_analyzer.models import User


def by_username(session: Session, username: str) -> User | None:
    return session.exec(select(User).where(User.username == username)).first()


def create(session: Session, *, username: str, name: str, password_hash: str) -> User:
    """Insert a user. The caller hashes the password — keeping the hash out of this
    module is what stops ``auth`` and ``queries.users`` importing each other.
    """
    user = User(username=username.strip(), name=name.strip(), password_hash=password_hash)
    session.add(user)
    session.commit()
    session.refresh(user)

    return user
