"""Database reads for the dashboard.

Every ``select()`` the dashboard runs lives here rather than in a route handler, so
the join shapes are reviewable in one place and the routers stay about HTTP.
"""
