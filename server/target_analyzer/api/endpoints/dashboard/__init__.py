"""The human dashboard: Trend, Session and Compare (implementation.md §10)."""

from fastapi import APIRouter, Depends

from target_analyzer.auth import require_user


def dashboard_router() -> APIRouter:
    """A router every dashboard module builds from, so the login gate is structural.

    The gate is the one thing a new view must not be able to forget, and a dependency
    spelled out once per module is one chance per module to omit it.
    """
    return APIRouter(prefix="/dashboard", tags=["dashboard"], dependencies=[Depends(require_user)])
