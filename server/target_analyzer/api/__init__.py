"""HTTP layer.

Route modules live in :mod:`target_analyzer.api.endpoints`. ``routers`` is the single
registration point: ``create_app`` just iterates it, so adding a surface is one line
here rather than another ``include_router`` in the app factory.
"""

from target_analyzer.api.endpoints import ingest
from target_analyzer.api.endpoints.core import auth, health
from target_analyzer.api.endpoints.dashboard import compare, image, session, trend

routers = (
    health.router,
    auth.router,
    ingest.router,
    trend.router,
    session.router,
    compare.router,
    image.router,
)
