"""HTTP layer.

Route modules live in :mod:`target_analyzer.api.endpoints`. ``routers`` is the single
registration point: ``create_app`` just iterates it, so adding a surface is one line
here rather than another ``include_router`` in the app factory.
"""

from target_analyzer.api.endpoints import ingest
from target_analyzer.api.endpoints.core import health

# The dashboard routers (Trend / Session / Compare) join this list in P5.
routers = (
    health.router,
    ingest.router,
)
