"""Registration: warping a photo of the target into the profile's canonical square.
See internal_docs/implementation.md §1 (the frame contract) and §9.

The fit is over-determined on purpose. Every detected marker contributes four corners,
so a board of four markers gives sixteen correspondences, and the reprojection residual
of that fit is a real confidence number. A four-point fit reproduces its own inputs
exactly and its residual is always zero, which says nothing at all — hence
:data:`MIN_MARKERS` and the ``None`` residual on the hand-clicked path.

:func:`register` is pure and headless so the self-check can drive it; the window lives
in :func:`register_interactive`.
"""

import math
import sys
from dataclasses import dataclass, replace
from typing import Literal

import cv2
import numpy as np

from ta_client import pick
from ta_client.board import aruco_dictionary, draw_rings
from ta_shared.profile import TargetProfile, mm_per_px

# Below three the fit is not over-determined enough for its residual to mean anything:
# two markers of a corner board span one edge, and one spans no area at all.
MIN_MARKERS = 3
# In canonical pixels, the destination space: sub-pixel corners land well under one.
RANSAC_REPROJ_PX = 3.0
SUPERSAMPLE = 2
# A sheet partly out of frame should read as paper, not as the default black void.
PAPER = (255, 255, 255)
# Millimetres on the target, so one number means the same for every profile. It measures
# the markers against each other, never against the print — refine_to_rings does that.
MAX_RESIDUAL_MM = 1.0
# Past these a ring fit is a misread circle rather than a mis-measured profile.
MAX_RING_SCALE = 0.12
MAX_RING_SHIFT = 0.05
MIN_RING_POINTS = 40
# Grey levels across the edge of the black. Paper texture and print noise sit far below.
MIN_EDGE_STEP = 4.0

# The order manual_corners_canon stores them in, which is the order a human must click.
CORNER_ORDER = "top-left, top-right, bottom-right, bottom-left"


class TooFewMarkers(ValueError):
    """Not enough of the board was found to fit a homography worth trusting."""


@dataclass(frozen=True)
class Registration:
    homography: np.ndarray
    normalized: np.ndarray
    residual: float | None
    n_markers: int
    n_inliers: int
    source: Literal["aruco", "manual"]
    # (x, y) stretch the printed rings asked for, or None when they could not be read.
    ring_correction: tuple[float, float] | None = None

    def params(self) -> dict:
        """Provenance for the payload — plain Python, no numpy scalars or arrays."""
        return {
            "homography": self.homography.tolist(),
            "registration_residual": self.residual,
            "registration_source": self.source,
            "n_markers": self.n_markers,
            "n_inliers": self.n_inliers,
            # params is a bare dict, so nothing coerces a tuple back on the way in.
            "ring_correction": None if self.ring_correction is None else list(self.ring_correction),
        }

    def summary(self, profile: TargetProfile) -> str:
        """One plain sentence for the person looking at the overlay."""
        if self.source == "manual":
            how = "registered from your 4 clicks"
        else:
            off = residual_mm(self, profile)
            agreement = "" if math.isinf(off) else f", agreeing to {off:.2f} mm"
            how = f"registered from {self.n_markers} markers{agreement}"

        if self.ring_correction is None:
            fit = "printed rings not read, so this rests on the profile's measurements"
        else:
            x, y = (100 * (s - 1) for s in self.ring_correction)
            fit = f"frame settled on the printed rings ({x:+.1f}% / {y:+.1f}%)"

        return f"{how} · {fit}"


def detect_board(photo: np.ndarray, profile: TargetProfile) -> tuple[np.ndarray, np.ndarray, int]:
    """Correspondences from every detected marker corner: ``(photo px, canonical px, n)``."""
    parameters = cv2.aruco.DetectorParameters()
    # Unrefined corners are whole-pixel, and the residual would measure that, not the fit.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # ponytail: stock thresholding. adaptiveThreshWinSizeMax is the knob to reach for if
    # detection fails on a 12 MP photo, where a marker is large next to the default window.
    detector = cv2.aruco.ArucoDetector(aruco_dictionary(profile.board.aruco_dict), parameters)
    corners, ids, _rejected = detector.detectMarkers(photo)

    src: list[list[float]] = []
    dst: list[tuple[float, float]] = []

    detected = [] if ids is None else zip(ids.flatten().tolist(), corners, strict=True)

    for marker_id, quad in detected:
        canonical = profile.board.markers.get(marker_id)

        if canonical is None:  # some other sheet's marker wandered into the frame
            continue

        # Clockwise from each marker's own top-left, which is the order the profile stores
        # — that is what registers a sideways photo correctly. Never sort these.
        src.extend(quad.reshape(4, 2).tolist())
        dst.extend(canonical)

    src_px = np.asarray(src, np.float32).reshape(-1, 1, 2)
    dst_px = np.asarray(dst, np.float32).reshape(-1, 1, 2)

    return src_px, dst_px, len(src) // 4


def fit_homography(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, int]:
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_PX)

    if homography is None:
        raise TooFewMarkers("the marker correspondences did not yield a homography")

    projected = cv2.perspectiveTransform(src, homography).reshape(-1, 2)
    # Every correspondence, not just the inliers: a marker RANSAC dropped is exactly what
    # the person confirming the warp needs to see.
    errors = np.sum((projected - dst.reshape(-1, 2)) ** 2, axis=1)

    return homography, float(np.sqrt(np.mean(errors))), int(mask.sum())


def warp(photo: np.ndarray, homography: np.ndarray, canon_size_px: int) -> np.ndarray:
    """Render the photo into the canonical square, exactly ``canon_size_px`` on a side."""
    dense = canon_size_px * SUPERSAMPLE
    upscale = np.diag([float(SUPERSAMPLE), float(SUPERSAMPLE), 1.0])
    # warpPerspective has no area-averaging mode, and bilinear straight from 12 MP aliases
    # the rings that holes are about to be clicked against.
    large = cv2.warpPerspective(photo, upscale @ homography, (dense, dense), borderValue=PAPER)

    return cv2.resize(large, (canon_size_px, canon_size_px), interpolation=cv2.INTER_AREA)


def _black_edge(normalized: np.ndarray, profile: TargetProfile) -> np.ndarray | None:
    """Points on the outer edge of the black, found as the steepest dark-to-light step."""
    if profile.black_ring is None:
        return None

    expected = profile.ring_radius_px(profile.black_ring)
    centre = profile.canon_size_px / 2
    grey = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    radii = np.arange(expected * 0.75, expected * 1.25, 0.5, dtype=np.float32)
    found = []

    for degrees in range(0, 360, 3):
        angle = math.radians(degrees)
        xs = (centre + radii * math.cos(angle)).reshape(-1, 1)
        ys = (centre + radii * math.sin(angle)).reshape(-1, 1)
        line = cv2.remap(grey, xs, ys, cv2.INTER_LINEAR)
        step = np.diff(cv2.GaussianBlur(line, (1, 9), 0).ravel())
        best = int(np.argmax(step))

        if step[best] > MIN_EDGE_STEP:
            found.append((float(xs[best, 0]), float(ys[best, 0]), float(radii[best])))

    if len(found) < MIN_RING_POINTS:
        return None

    points = np.asarray(found)
    # A patch or a shot-out edge puts a ray somewhere else entirely; the rest agree.
    spread = np.abs(points[:, 2] - np.median(points[:, 2]))

    return points[spread < 2.0 * (np.median(spread) + 1.0), :2]


def _axis_aligned_ellipse(points: np.ndarray) -> tuple[float, float, float, float] | None:
    """``(cx, cy, a, b)`` of ((x-cx)/a)^2 + ((y-cy)/b)^2 = 1, by least squares.

    Axis-aligned rather than free: the error being measured is a per-axis stretch of the
    canonical frame, and a tilted fit would absorb noise into a rotation that the
    correction below cannot express anyway.
    """
    # Fit about the points' own centroid: on raw canonical coordinates the squared terms
    # run to 10^6 and the normal equations lose the fit entirely.
    origin = points.mean(axis=0)
    x, y = (points - origin).T
    design = np.stack([x * x, x, y * y, y], axis=1)
    (a, b, c, d), *_ = np.linalg.lstsq(design, np.ones(len(x)), rcond=None)

    if a <= 0 or c <= 0:
        return None

    cx, cy = -b / (2 * a), -d / (2 * c)
    scale = 1 + a * cx * cx + c * cy * cy

    if scale <= 0:
        return None

    return cx + origin[0], cy + origin[1], math.sqrt(scale / a), math.sqrt(scale / c)


def refine_to_rings(
    photo: np.ndarray, registration: Registration, profile: TargetProfile
) -> Registration:
    """Settle the frame against the printed rings; unchanged if they cannot be read.

    The corners are only as good as the profile's idea of where they sit, and that comes
    off a tape measure. The print does not: ring spacing is whatever the press set,
    identically on every card. A wrong corner spacing stretches the frame by a constant
    factor per axis, and fitting one ring of known radius measures that exactly.
    """
    # ponytail: anchored on the black alone, which assumes its edge falls exactly on ring
    # 7 — measured 1.1-1.4% out. Fitting every ring line instead would drop the assumption.
    points = _black_edge(registration.normalized, profile)

    if points is None or len(points) < MIN_RING_POINTS:
        return registration

    fitted = _axis_aligned_ellipse(points)

    if fitted is None:
        return registration

    cx, cy, semi_x, semi_y = fitted
    centre = profile.canon_size_px / 2
    expected = profile.ring_radius_px(profile.black_ring)
    stretch = (expected / semi_x, expected / semi_y)
    shift = max(abs(cx - centre), abs(cy - centre)) / profile.canon_size_px

    if max(abs(s - 1) for s in stretch) > MAX_RING_SCALE or shift > MAX_RING_SHIFT:
        return registration  # not measurement error — something was misread

    correction = np.array(
        [
            [stretch[0], 0.0, centre - stretch[0] * cx],
            [0.0, stretch[1], centre - stretch[1] * cy],
            [0.0, 0.0, 1.0],
        ]
    )
    homography = correction @ registration.homography

    return replace(
        registration,
        homography=homography,
        normalized=warp(photo, homography, profile.canon_size_px),
        ring_correction=stretch,
    )


def register(photo: np.ndarray, profile: TargetProfile) -> Registration:
    """Register by marker detection alone. Pure: no window, no input."""
    src, dst, n_markers = detect_board(photo, profile)

    if n_markers < MIN_MARKERS:
        raise TooFewMarkers(f"found {n_markers} of the board's markers, need {MIN_MARKERS}")

    homography, residual, n_inliers = fit_homography(src, dst)

    return Registration(
        homography=homography,
        normalized=warp(photo, homography, profile.canon_size_px),
        residual=residual,
        n_markers=n_markers,
        n_inliers=n_inliers,
        source="aruco",
    )


def register_manual(photo: np.ndarray, profile: TargetProfile) -> Registration | None:
    """The always-works floor: click the four corners of the target. ``None`` if cancelled."""
    title = f"Click the 4 corners, in this order: {CORNER_ORDER}"
    dst = np.asarray(profile.manual_corners_canon, np.float32).reshape(-1, 1, 2)

    while True:
        corners = pick.pick_points(photo, title, exact=4)

        if corners is None:
            return None

        src = np.asarray(corners, np.float32).reshape(-1, 1, 2)
        homography, _mask = cv2.findHomography(src, dst, 0)  # exact four-point fit

        if homography is not None:
            break

        # A slip of the mouse, not a reason to throw away everything clicked so far.
        print("those 4 corners are collinear — click them again", file=sys.stderr)

    return Registration(
        homography=homography,
        normalized=warp(photo, homography, profile.canon_size_px),
        residual=None,
        n_markers=0,
        n_inliers=4,
        source="manual",
    )


def residual_mm(registration: Registration, profile: TargetProfile) -> float:
    """The registration's residual in millimetres on the target, or infinity if it has none."""
    scale = mm_per_px(profile)

    if registration.residual is None or scale is None:
        return math.inf

    return registration.residual * scale


def register_interactive(photo: np.ndarray, profile: TargetProfile) -> Registration | None:
    """Register, asking for help only when the fit cannot vouch for itself. ``None`` aborts."""
    registration = None

    try:
        registration = refine_to_rings(photo, register(photo, profile), profile)
    except TooFewMarkers as exc:
        print(f"{exc} — clicking the corners instead", file=sys.stderr)

    # A profile with no black aiming area offers nothing to check the frame against, and
    # that is the printed sheet, whose geometry is exact by construction.
    verified = registration is not None and (
        registration.ring_correction is not None or profile.black_ring is None
    )

    if verified and residual_mm(registration, profile) <= MAX_RESIDUAL_MM:
        return registration

    while True:
        if registration is None:
            registration = register_manual(photo, profile)

            if registration is None:
                return None

            registration = refine_to_rings(photo, registration, profile)

        # Asking about the warp, not the clicked positions: it catches corners clicked in
        # the wrong order, which their positions alone do not.
        lines = [
            "Do the orange rings sit on the printed ones?",
            registration.summary(profile),
            "[Enter] yes, on to the holes      [any other key] click the 4 corners again",
        ]
        overlay = draw_rings(registration.normalized, profile)

        if pick.confirm(overlay, "Confirm registration", lines):
            return registration

        registration = None
