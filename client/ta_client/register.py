"""Registration: warping a photo of the target into the profile's canonical square.
See implementation.md §1 (the frame contract) and §9.

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
from ta_client.marks import MIN_SCORE, find_marks
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
# Grey levels between a printed ring line and the paper either side of it.
MIN_LINE_CONTRAST = 25.0
RING_RAYS = 120
# Fewer read than this and the rings between them are guessed rather than measured.
MIN_RING_LINES = 4

# The order manual_corners_canon stores them in, which is the order a human must click.
CORNER_ORDER = "top-left, top-right, bottom-right, bottom-left"

Source = Literal["aruco", "marks", "manual"]


class TooFewMarkers(ValueError):
    """Not enough of the board was found to fit a homography worth trusting."""


@dataclass(frozen=True)
class Registration:
    homography: np.ndarray
    normalized: np.ndarray
    residual: float | None
    n_markers: int
    n_inliers: int
    source: Source
    # (x, y) stretch the printed rings asked for, or None when they could not be read.
    ring_correction: tuple[float, float] | None = None
    # (profile radius, radius the line was read at) per printed ring, centre first: the
    # radial remap that lands each line on its ring. None when too few lines were read.
    radial: tuple[tuple[float, float], ...] | None = None

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
            # With this set, the frame is the homography and then this remap: the homography
            # alone lands a point up to 2.5 % off at the outer ring.
            "radial_correction": None if self.radial is None else [list(k) for k in self.radial],
        }

    def summary(self, profile: TargetProfile) -> str:
        """One plain sentence for the person looking at the overlay."""
        if self.source == "manual":
            how = "registered from your 4 clicks"
        elif self.source == "marks":
            how = "registered from the printed corner marks"
        else:
            off = residual_mm(self, profile)
            agreement = "" if math.isinf(off) else f", agreeing to {off:.2f} mm"
            how = f"registered from {self.n_markers} markers{agreement}"

        if self.ring_correction is None:
            fit = "printed rings not read, so this rests on the profile's measurements"
        else:
            x, y = (100 * (s - 1) for s in self.ring_correction)
            fit = f"frame settled on the printed rings ({x:+.1f}% / {y:+.1f}%)"

            if self.radial is not None:
                fit += f", {len(self.radial) - 1} ring lines fitted"

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


def warp(
    photo: np.ndarray,
    homography: np.ndarray,
    canon_size_px: int,
    radial: tuple[tuple[float, float], ...] | None = None,
) -> np.ndarray:
    """Render the photo into the canonical square, exactly ``canon_size_px`` on a side."""
    dense = canon_size_px * SUPERSAMPLE
    upscale = np.diag([float(SUPERSAMPLE), float(SUPERSAMPLE), 1.0])

    if radial is None:
        # warpPerspective has no area-averaging mode, and bilinear straight from 12 MP
        # aliases the rings that holes are about to be clicked against.
        large = cv2.warpPerspective(photo, upscale @ homography, (dense, dense), borderValue=PAPER)
    else:
        # Sampled straight from the photo rather than re-warping a rendered frame, which
        # would blur it twice.
        centre = canon_size_px / 2
        grid = np.mgrid[0:dense, 0:dense].astype(np.float32) / SUPERSAMPLE - centre
        radius = np.hypot(grid[0], grid[1])
        ring, read = (np.asarray(k, np.float32) for k in zip(*radial, strict=True))
        # Past the outermost line the last ring's ratio carries on, out to the corners.
        source = np.where(
            radius <= ring[-1], np.interp(radius, ring, read), radius * read[-1] / ring[-1]
        )
        ratio = np.divide(source, radius, out=np.ones_like(radius), where=radius > 0)
        points = np.stack([grid[1] * ratio + centre, grid[0] * ratio + centre], axis=-1)
        mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), np.linalg.inv(homography))
        mapped = mapped.reshape(dense, dense, 2)
        large = cv2.remap(
            photo, mapped[..., 0], mapped[..., 1], cv2.INTER_LINEAR, borderValue=PAPER
        )

    return cv2.resize(large, (canon_size_px, canon_size_px), interpolation=cv2.INTER_AREA)


def _rays(grey: np.ndarray, centre: float, angles: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Grey levels along rays from the centre: one row per angle, one column per radius."""
    xs = (centre + np.outer(np.cos(angles), radii)).astype(np.float32)
    ys = (centre + np.outer(np.sin(angles), radii)).astype(np.float32)

    return cv2.remap(grey, xs, ys, cv2.INTER_LINEAR)


def _ring_lines(normalized: np.ndarray, profile: TargetProfile) -> dict[int, float]:
    """The median radius each printed ring line was read at, except the black's own edge."""
    grey = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    centre = profile.canon_size_px / 2
    black = profile.ring_radius_px(profile.black_ring)
    # Short of halfway to the next line, so a ray never lands on a neighbour.
    reach = 0.4 * min(np.diff(sorted(profile.ring_radii_px)))
    angles = np.radians(np.arange(RING_RAYS) * 360 / RING_RAYS)
    found = {}

    for ring in range(1, profile.n_rings + 1):
        if ring == profile.black_ring:
            continue

        expected = profile.ring_radius_px(ring)
        radii = np.arange(expected - reach, expected + reach, 0.25, dtype=np.float32)
        rays = cv2.GaussianBlur(_rays(grey, centre, angles, radii), (5, 1), 0)
        # Dark lines on the paper, pale ones on the black.
        rays = rays if expected > black else 255 - rays
        contrast = np.median(rays, axis=1) - rays.min(axis=1)
        read = radii[rays.argmin(axis=1)][contrast > MIN_LINE_CONTRAST]

        # Digits, holes and patches take some rays; the median of the rest is the line.
        if len(read) >= MIN_RING_POINTS:
            found[ring] = float(np.median(read))

    return found


def _radial(
    normalized: np.ndarray, profile: TargetProfile
) -> tuple[tuple[float, float], ...] | None:
    """``(profile radius, read radius)`` pairs from the centre out, or ``None``."""
    found = _ring_lines(normalized, profile)

    if len(found) < MIN_RING_LINES:
        return None

    pairs = sorted((float(profile.ring_radius_px(ring)), read) for ring, read in found.items())
    reads = [read for _, read in pairs]

    # A misread line out of order, or far off its ring, is not a measurement of this sheet.
    if reads != sorted(reads) or any(abs(read / ring - 1) > MAX_RING_SCALE for ring, read in pairs):
        return None

    return ((0.0, 0.0), *pairs)


def _black_edge(normalized: np.ndarray, profile: TargetProfile) -> np.ndarray | None:
    """Points on the outer edge of the black, found as the steepest dark-to-light step."""
    if profile.black_ring is None:
        return None

    expected = profile.ring_radius_px(profile.black_ring)
    centre = profile.canon_size_px / 2
    grey = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    radii = np.arange(expected * 0.75, expected * 1.25, 0.5, dtype=np.float32)
    angles = np.radians(np.arange(0, 360, 3))
    step = np.diff(cv2.GaussianBlur(_rays(grey, centre, angles, radii), (9, 1), 0), axis=1)
    best = step.argmax(axis=1)
    keep = step[np.arange(len(angles)), best] > MIN_EDGE_STEP

    if keep.sum() < MIN_RING_POINTS:
        return None

    radius, angle = radii[best][keep], angles[keep]
    points = np.stack(
        [centre + radius * np.cos(angle), centre + radius * np.sin(angle), radius], axis=1
    )
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

    The corners come off a tape measure, and a wrong corner spacing stretches the frame by
    a constant factor per axis, which the black's edge measures. The lines are then read
    one by one and the frame bent radially onto them: they are what a shot is scored
    against, and they drift from the profile by up to 2.5 % at the outer ring, by a
    different amount on each photo.
    """
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
    normalized = warp(photo, homography, profile.canon_size_px)
    # ponytail: one radius per line, for every direction. A sheet that curls unevenly keeps
    # about 1 mm at the outer ring; reading the lines per sector is the upgrade.
    radial = _radial(normalized, profile)

    if radial is not None:
        normalized = warp(photo, homography, profile.canon_size_px, radial)

    return replace(
        registration,
        homography=homography,
        normalized=normalized,
        ring_correction=stretch,
        radial=radial,
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


def from_corners(
    photo: np.ndarray, corners: list[tuple[float, float]], profile: TargetProfile, source: Source
) -> Registration | None:
    """The exact four-point fit onto ``manual_corners_canon``; ``None`` if they are collinear."""
    src = np.asarray(corners, np.float32).reshape(-1, 1, 2)
    dst = np.asarray(profile.manual_corners_canon, np.float32).reshape(-1, 1, 2)
    homography, _mask = cv2.findHomography(src, dst, 0)

    if homography is None:
        return None

    return Registration(
        homography=homography,
        normalized=warp(photo, homography, profile.canon_size_px),
        residual=None,
        n_markers=0,
        n_inliers=4,
        source=source,
    )


def register_manual(
    photo: np.ndarray, profile: TargetProfile, initial: list[tuple[float, float]] | None = None
) -> Registration | None:
    """The always-works floor: click the four corners of the target. ``None`` if cancelled."""
    title = f"Click the 4 corners, in this order: {CORNER_ORDER}"

    while True:
        corners = pick.pick_points(photo, title, exact=4, initial=initial)

        if corners is None:
            return None

        registration = from_corners(photo, corners, profile, "manual")

        if registration is not None:
            return registration

        # A slip of the mouse, not a reason to throw away everything clicked so far.
        print("those 4 corners are collinear — click them again", file=sys.stderr)
        initial = corners


def residual_mm(registration: Registration, profile: TargetProfile) -> float:
    """The registration's residual in millimetres on the target, or infinity if it has none."""
    scale = mm_per_px(profile)

    if registration.residual is None or scale is None:
        return math.inf

    return registration.residual * scale


def register_interactive(photo: np.ndarray, profile: TargetProfile) -> Registration | None:
    """Register, asking for help only when the fit cannot vouch for itself. ``None`` aborts."""
    registration = None
    seed = None

    try:
        registration = refine_to_rings(photo, register(photo, profile), profile)
    except TooFewMarkers as exc:
        marks = find_marks(photo, profile)

        # Seeded only when some mark was clearly found: a sheet where none was — tilted, or
        # unmarked — would open holding four points in the wrong places.
        if marks is not None and max(marks.scores) >= MIN_SCORE:
            seed = marks.corners

        if marks is not None and marks.confident:
            registration = from_corners(photo, marks.corners, profile, "marks")

        if registration is None:
            unclear = "the printed corner marks are unclear — place the corners"
            print(f"{exc}, and {unclear}", file=sys.stderr)
        else:
            registration = refine_to_rings(photo, registration, profile)

    # A profile with no black aiming area offers nothing to check the frame against, and
    # that is the printed sheet, whose geometry is exact by construction.
    verified = registration is not None and (
        registration.ring_correction is not None or profile.black_ring is None
    )

    if verified and residual_mm(registration, profile) <= MAX_RESIDUAL_MM:
        return registration

    while True:
        if registration is None:
            registration = register_manual(photo, profile, seed)

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
