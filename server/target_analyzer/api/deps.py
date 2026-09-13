"""Shared FastAPI dependencies as typed aliases, to keep handler signatures lean.

Note: an ``Annotated[...]`` dependency has no default value, so a parameter typed with
one of these must come **before** any parameter that does have one (Form/File/Query).
"""

from typing import Annotated

from fastapi import Depends
from sqlmodel import Session

from target_analyzer.db import get_session

DbSession = Annotated[Session, Depends(get_session)]
