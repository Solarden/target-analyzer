"""Mint a machine bearer token for the Mac client: ``python -m target_analyzer.create_token``.

Prints both halves once. The token goes in the Mac's config; only its sha256 is ever
stored server-side, so losing the token means minting a new one, not recovering this.
"""

import hashlib
import secrets


def main() -> None:
    token = secrets.token_urlsafe(32)
    digest = hashlib.sha256(token.encode()).hexdigest()

    print("Mac client   TA_INGEST_TOKEN      =", token)
    print("Pi server    TA_INGEST_TOKEN_HASH =", digest)
    print("\nStore the token in a password manager — the server keeps only the hash.")


if __name__ == "__main__":
    main()
