"""Jinja2 setup for the dashboard.

Server-rendered and deliberately plain. The one filter worth having is the metric
formatter: almost every number on the dashboard is a millimetre measurement that is
None on a profile with no physical diameter, and repeating that fallback in every
template is how one of them ends up printing "None".
"""

from datetime import date, datetime
from pathlib import Path

from fastapi.templating import Jinja2Templates

from target_analyzer.config import get_settings

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def format_mm(value: float | None) -> str:
    """A millimetre measurement, or an em dash when the profile carries no scale."""
    if value is None:
        return "—"

    return f"{value:.1f} mm"


def format_number(value: float | None, digits: int) -> str:
    if value is None:
        return "—"

    return f"{value:.{digits}f}"


def format_metres(value: float | None) -> str:
    """A distance with no trailing zeros: 25.0 -> "25", 12.5 -> "12.5".

    Rounding to whole metres would print a 12.5 m range as "12 m", and two ranges a
    decimetre apart as the same filter option.
    """
    if value is None:
        return "—"

    return f"{value:g}"


def ring_class(ring: int, n_rings: int) -> str:
    """The class a hole marker is drawn with, banded by how well the shot scored.

    A component class, not a Tailwind utility: a utility name assembled at runtime
    appears in no template for the ``@source`` scanner to find, so it would be purged
    from the stylesheet and every marker would ship colourless, with nothing warning.
    """
    if ring <= 0:
        return "hole-miss"

    if ring >= n_rings - 1:
        return "hole-good"

    if ring >= n_rings - 4:
        return "hole-ok"

    return "hole-poor"


def format_shooter(value: str | None) -> str:
    return value or "You"


def format_day(value: date | datetime | None) -> str:
    if value is None:
        return "—"

    return value.strftime("%Y-%m-%d")


templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
templates.env.filters["mm"] = format_mm
templates.env.filters["num"] = format_number
templates.env.filters["day"] = format_day
templates.env.filters["shooter"] = format_shooter
templates.env.filters["m"] = format_metres
templates.env.filters["ring_class"] = ring_class
# Reachable from every template without threading it through each handler: the
# footer link the AGPL asks a network-served app to offer its users.
templates.env.globals["source_url"] = get_settings().source_url
