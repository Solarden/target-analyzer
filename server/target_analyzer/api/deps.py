"""Shared FastAPI dependencies as typed aliases, to keep handler signatures lean.

Note: an ``Annotated[...]`` dependency has no default value, so a parameter typed with
one of these must come **before** any parameter that does have one (Form/File/Query).
"""

from typing import Annotated

from fastapi import Depends
from sqlmodel import Session

from target_analyzer.auth import require_user
from target_analyzer.db import get_session
from target_analyzer.models import User

DbSession = Annotated[Session, Depends(get_session)]
CurrentUser = Annotated[User, Depends(require_user)]
