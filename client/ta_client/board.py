"""The marker board: resolving its ArUco dictionary, rendering it, and drawing the
target it belongs to. See internal_docs/implementation.md §4 and §9.

Two renderings, because they answer different questions. :func:`render_svg` is the
sheet you print — SVG carries millimetres, so it comes off the printer at the exact
physical size the profile describes, with no DPI arithmetic. :func:`render_markers`
is the raster board the self-check warps into a synthetic photo.

Run ``python -m ta_client.board profiles/issf_precision.json`` to produce the sheet.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from ta_shared.profile import TargetProfile, load_profile, mm_per_px

# Orange reads on both the white paper and the black of a printed bull.
RING_COLOUR = (0, 140, 255)
SHEET_MARGIN_MM = 10.0
# Printed next to a caption so a ruler can catch a printer that quietly scaled the page.
CALIBRATION_MM = 50.0


def aruco_dictionary(name: str) -> cv2.aruco.Dictionary:
    """Resolve a profile's ``aruco_dict`` string to the cv2 dictionary it names."""
    # The DICT_ prefix keeps an unguarded getattr on profile-supplied text from reaching
    # something like __loader__ and failing incoherently instead of loudly.
    constant = getattr(cv2.aruco, name, None) if name.startswith("DICT_") else None

    if constant is None:
        raise ValueError(f"unknown ArUco dictionary: {name!r}")

    return cv2.aruco.getPredefinedDictionary(constant)


def marker_bits(dictionary: cv2.aruco.Dictionary, marker_id: int) -> np.ndarray:
    """The marker's module grid as a boolean array, True where a module is black.

    Asking for one pixel per module is the smallest image the generator will draw, so
    the result is the pattern itself rather than a rendering of it.
    """
    modules = dictionary.markerSize + 2  # the quiet border the generator draws around it

    return cv2.aruco.generateImageMarker(dictionary, marker_id, modules) == 0


def _canonical_rect(corners: list[tuple[float, float]]) -> tuple[int, int, int, int]:
    """The marker's ``(x0, y0, x1, y1)`` box in canonical px.

    A quad that is not an axis-aligned square would print a marker the detector then
    reads at the wrong orientation or aspect — wrong silently, on paper, so it is worth
    refusing here rather than discovering it on a photographed sheet.
    """
    xs = {x for x, _ in corners}
    ys = {y for _, y in corners}

    if len(xs) != 2 or len(ys) != 2:
        raise ValueError(f"marker quad is not axis-aligned: {corners}")

    (x0, x1), (y0, y1) = sorted(xs), sorted(ys)

    if x1 - x0 != y1 - y0:
        raise ValueError(f"marker quad is not square: {corners}")

    return int(x0), int(y0), int(x1), int(y1)


def render_markers(profile: TargetProfile, *, scale: int = 1) -> np.ndarray:
    """The board alone, on a white canonical canvas — markers, no rings."""
    dictionary = aruco_dictionary(profile.board.aruco_dict)
    side = profile.canon_size_px * scale
    canvas = np.full((side, side, 3), 255, np.uint8)
    modules = dictionary.markerSize + 2

    for marker_id, corners in profile.board.markers.items():
        x0, y0, x1, y1 = _canonical_rect(corners)
        size = (x1 - x0) * scale
        # Generate at a whole number of pixels per module and area-average down: asking
        # for 40 px of a 6-module marker directly gives modules 6.67 px wide, i.e. uneven.
        square = cv2.aruco.generateImageMarker(dictionary, marker_id, modules * -(-size // modules))
        marker = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
        canvas[y0 * scale : y0 * scale + size, x0 * scale : x0 * scale + size] = marker[:, :, None]

    return canvas


def draw_rings(image: np.ndarray, profile: TargetProfile) -> np.ndarray:
    """A copy of a canonical-frame image with the profile's rings and centre drawn on.

    Used twice: to confirm a registration (drawn rings landing on printed rings is a
    direct read of whether the frame is right) and as the backdrop for clicking holes.
    """
    out = image.copy()
    centre = (profile.canon_size_px // 2, profile.canon_size_px // 2)

    for radius in profile.ring_radii_px:
        cv2.circle(out, centre, radius, RING_COLOUR, 1, cv2.LINE_AA)

    cv2.drawMarker(out, centre, RING_COLOUR, cv2.MARKER_CROSS, 24, 1)

    return out


def render_svg(profile: TargetProfile) -> str:
    """The printable target: rings, markers, frame outline, and a calibration line."""
    if profile.target_diam_mm is None:
        raise ValueError(
            f"{profile.name} v{profile.version} has no target_diam_mm, so the sheet has no "
            "physical scale to print at"
        )

    scale = mm_per_px(profile)
    side = profile.canon_size_px * scale
    centre = side / 2
    width = side + 2 * SHEET_MARGIN_MM
    footer = 14.0
    dictionary = aruco_dictionary(profile.board.aruco_dict)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.3f}mm" '
        f'height="{width + footer:.3f}mm" viewBox="0 0 {width:.3f} {width + footer:.3f}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<g transform="translate({SHEET_MARGIN_MM},{SHEET_MARGIN_MM})">',
        # The canonical frame edge: the four points the manual fallback asks you to click.
        f'<rect x="0" y="0" width="{side:.3f}" height="{side:.3f}" fill="none" '
        f'stroke="#bbb" stroke-width="0.15" stroke-dasharray="2 2"/>',
    ]

    for radius in profile.ring_radii_px:
        parts.append(
            f'<circle cx="{centre:.3f}" cy="{centre:.3f}" r="{radius * scale:.3f}" '
            f'fill="none" stroke="#000" stroke-width="0.2"/>'
        )

    for marker_id, corners in profile.board.markers.items():
        x0, y0, x1, _ = _canonical_rect(corners)
        bits = marker_bits(dictionary, marker_id)
        cell = (x1 - x0) * scale / len(bits)
        parts.append(f'<g transform="translate({x0 * scale:.3f},{y0 * scale:.3f})">')
        parts += [
            f'<rect x="{col * cell:.3f}" y="{row * cell:.3f}" width="{cell:.3f}" '
            f'height="{cell:.3f}" fill="#000"/>'
            for row, col in zip(*np.nonzero(bits), strict=True)
        ]
        parts.append("</g>")

    parts.append("</g>")
    base = width + footer - 5
    parts += [
        f'<line x1="{SHEET_MARGIN_MM}" y1="{base:.3f}" x2="{SHEET_MARGIN_MM + CALIBRATION_MM}" '
        f'y2="{base:.3f}" stroke="#000" stroke-width="0.3"/>',
        f'<text x="{SHEET_MARGIN_MM + CALIBRATION_MM + 3}" y="{base + 1.2:.3f}" '
        f'font-family="sans-serif" font-size="3">'
        f"{profile.name} v{profile.version} — print at 100%; this line must measure "
        f"{CALIBRATION_MM:.0f} mm</text>",
        "</svg>",
    ]

    return "\n".join(parts)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Render a printable target sheet from a profile.")
    parser.add_argument("profile", type=Path, help="path to a profile JSON")
    parser.add_argument("--out", type=Path, help="output .svg (default: <name>_v<version>.svg)")
    args = parser.parse_args(argv)

    profile = load_profile(args.profile)
    out = args.out or Path(f"{profile.name}_v{profile.version}.svg")
    out.write_text(render_svg(profile), encoding="utf-8")
    side = profile.canon_size_px * mm_per_px(profile)
    print(f"wrote {out} — canonical square is {side:.1f} mm; print at 100%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
