"""Create a dashboard login: ``python -m target_analyzer.create_user``.

There is no public registration — a household app with one target and one shooter —
so every login is made here.

    python -m target_analyzer.create_user --username alice --name "Alice"
"""

import argparse
import getpass
import sys

from sqlmodel import Session

from target_analyzer.auth import hash_password
from target_analyzer.db import get_engine
from target_analyzer.queries import users


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a target-analyzer dashboard login.")
    parser.add_argument("--username", required=True, help="login handle (unique)")
    parser.add_argument("--name", help="display name (defaults to the username)")
    args = parser.parse_args()
    username = args.username.strip()
    password = getpass.getpass("Password: ")

    if not password:
        sys.exit("aborted: empty password")

    if password != getpass.getpass("Confirm password: "):
        sys.exit("aborted: passwords do not match")

    with Session(get_engine()) as session:
        if users.by_username(session, username) is not None:
            sys.exit(f"error: username {username!r} already exists")

        user = users.create(
            session,
            username=username,
            name=(args.name or username),
            password_hash=hash_password(password),
        )

    print(f"Created user {user.username!r} (id={user.id}).")


if __name__ == "__main__":
    main()
