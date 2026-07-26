"""Machine authentication for the ingest API (implementation.md §8).

Separate from the human dashboard's cookie session (P5) on purpose: the Mac client is
not a browser, so it gets a bearer token and nothing else.

The token is high-entropy (``secrets.token_urlsafe(32)``), which is why the stored
credential is a plain sha256 and not bcrypt: bcrypt's work factor buys resistance to
guessing a *human-chosen* secret, and 256 bits of randomness has nothing to guess. What
does matter is comparing in constant time — hence ``hmac.compare_digest``.
"""

import hashlib
import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from target_analyzer.config import get_settings

# auto_error=False so a missing header reaches us as None and gets the same 401 as a
# wrong one — FastAPI's own default would raise a 403 instead, which reads as "your
# token is fine but you lack rights" and would send the Mac client down a wrong path.
_bearer = HTTPBearer(auto_error=False)


def require_machine(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Reject anything that is not the configured machine token.

    503 (not 401) when no hash is configured: the caller's credentials are not the
    problem, the server is unfinished, and a 401 would send the Mac into a retry loop
    against a box that can never accept it.
    """
    expected = get_settings().ingest_token_hash

    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ingest is not configured (TA_INGEST_TOKEN_HASH unset)",
        )

    presented = credentials.credentials if credentials else ""
    digest = hashlib.sha256(presented.encode()).hexdigest()

    if not hmac.compare_digest(digest, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid machine token",
            headers={"WWW-Authenticate": "Bearer"},
        )
