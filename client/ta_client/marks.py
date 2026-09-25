"""Finding the four ⊕ corner marks printed on the target, whose cross centres are the points
``manual_corners_canon`` stores.

Two stages, because a search for circles near the photo's corners finds other sheets first:
the black aiming disc gives the target's centre and scale, and each mark is then matched in
a window around where the profile puts it. The matching runs in the canonical frame of the
current estimate, where a mark is a known shape at a known size, and runs twice more once
the first four points have removed most of the perspective.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ta_shared.profile import TargetProfile, mm_per_px

# The printed mark, measured on the corpus: a 100 mm circle round a 30 mm cross.
MARK_RADIUS_MM = 50.0
CROSS_HALF_MM = 15.0
LINE_MM = 1.5
# The disc search runs on a copy this size: the disc is a quarter of the sheet across, and
# full resolution only adds paper texture.
DISC_WORK_PX = 1000
# Search half-widths in canonical pixels. The first absorbs what the disc cannot say about
# perspective and in-plane rotation; the second only the first pass's own error.
COARSE_WINDOW = 160
FINE_WINDOW = 16
# Normalized cross-correlation against the rendered circle, measured on the corpus: at most
# 0.24 away from a mark, 0.58 or more on a whole one.
MIN_SCORE = 0.4
# The disc is round on paper; past this the photo is too oblique to register at all.
MIN_AXIS_RATIO = 0.6


@dataclass(frozen=True)
class Marks:
    # Photo pixels, in manual_corners_canon's order.
    corners: list[tuple[float, float]]
    scores: list[float]

    @property
    def confident(self) -> bool:
        return min(self.scores) >= MIN_SCORE


def _find_disc(grey: np.ndarray, profile: TargetProfile) -> np.ndarray | None:
    """The photo -> canonical homography the black disc alone implies, or ``None``.

    A circle says nothing about rotation, so this assumes the photo is upright, as its
    EXIF orientation already makes it: the fitted ellipse is taken as a stretch with no
    turn in it.
    """
    # ponytail: measured exact to ±8° of tilt; at 15° the marks score under MIN_SCORE and
    # the picker opens instead. A coarse pass at a few trial rotations is the upgrade.
    if profile.black_ring is None:
        return None

    scale = DISC_WORK_PX / max(grey.shape)
    small = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    _, dark = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    best = None

    for contour in contours:
        hull = cv2.convexHull(contour)

        if len(hull) < 5 or cv2.contourArea(hull) < (0.05 * DISC_WORK_PX) ** 2:
            continue

        (cx, cy), (w, h), angle = cv2.fitEllipse(hull)

        if min(w, h) < MIN_AXIS_RATIO * max(w, h) or min(w, h) < 0.05 * DISC_WORK_PX:
            continue

        ellipse_area = math.pi * w * h / 4
        hull_area = cv2.contourArea(hull)
        # The disc's hull is its ellipse: numbers, holes and patches are all inside it.
        # A dark background region is ragged and its hull fills its ellipse poorly.
        if abs(hull_area / ellipse_area - 1) > 0.05:
            continue

        if cv2.contourArea(contour) < 0.6 * hull_area:
            continue

        if best is None or ellipse_area > best[0]:
            best = (ellipse_area, cx, cy, w, h, angle)

    if best is None:
        return None

    _, cx, cy, w, h, angle = best
    radius = profile.ring_radius_px(profile.black_ring)
    turn = np.array(
        [
            [math.cos(math.radians(angle)), -math.sin(math.radians(angle))],
            [math.sin(math.radians(angle)), math.cos(math.radians(angle))],
        ]
    )
    # Canonical -> photo: a symmetric stretch mapping the disc onto the ellipse.
    stretch = turn @ np.diag([w / 2 / radius, h / 2 / radius]) @ turn.T / scale
    centre = profile.canon_size_px / 2
    to_photo = np.eye(3)
    to_photo[:2, :2] = stretch
    to_photo[:2, 2] = np.array([cx, cy]) / scale - stretch @ [centre, centre]

    return np.linalg.inv(to_photo)


def _line_px(profile: TargetProfile) -> int:
    return max(1, round(LINE_MM / mm_per_px(profile)))


def _template(profile: TargetProfile, *, circle: bool) -> np.ndarray:
    per_mm = 1 / mm_per_px(profile)
    radius = round(MARK_RADIUS_MM * per_mm)
    half = round(CROSS_HALF_MM * per_mm)
    line = _line_px(profile)
    side = 2 * ((radius if circle else half) + 2 * line) + 1
    mid = side // 2
    image = np.full((side, side), 255, np.uint8)

    if circle:
        cv2.circle(image, (mid, mid), radius, 0, line, cv2.LINE_AA)

    cv2.line(image, (mid - half, mid), (mid + half, mid), 0, line, cv2.LINE_AA)
    cv2.line(image, (mid, mid - half), (mid, mid + half), 0, line, cv2.LINE_AA)

    return _soften(image, line)


def _soften(image: np.ndarray, line: int) -> np.ndarray:
    # A line a few pixels wide correlates with nothing a few pixels off it, and the frame
    # the first pass searches is off by a few per cent in scale.
    return cv2.GaussianBlur(image, (0, 0), line)


def _vertex(before: float, at: float, after: float) -> float:
    """Offset of a parabola's vertex from the middle of three samples, 0 if it has none."""
    curve = before - 2 * at + after

    return 0.5 * (before - after) / curve if curve < 0 else 0.0


def _peak(scores: np.ndarray) -> tuple[float, float, float]:
    """``(x, y, score)`` of the best match, sub-pixel by a parabola through its neighbours."""
    _, best, _, (x, y) = cv2.minMaxLoc(scores)
    dx = _vertex(*scores[y, x - 1 : x + 2]) if 0 < x < scores.shape[1] - 1 else 0.0
    dy = _vertex(*scores[y - 1 : y + 2, x]) if 0 < y < scores.shape[0] - 1 else 0.0

    return x + dx, y + dy, float(best)


def _pass(
    grey: np.ndarray,
    to_canon: np.ndarray,
    profile: TargetProfile,
    template: np.ndarray,
    window: int,
) -> Marks:
    """Match each mark near where ``to_canon`` puts it; the result in photo pixels."""
    margin = window + template.shape[0]
    side = profile.canon_size_px + 2 * margin
    shift = np.array([[1, 0, margin], [0, 1, margin], [0, 0, 1]], float)
    canvas = cv2.warpPerspective(
        grey, shift @ to_canon, (side, side), flags=cv2.INTER_LINEAR, borderValue=255
    )
    canvas = _soften(canvas, _line_px(profile))
    to_photo = np.linalg.inv(shift @ to_canon)
    reach = window + template.shape[0] // 2
    corners, scores = [], []

    for x, y in profile.manual_corners_canon:
        cx, cy = round(x) + margin, round(y) + margin
        area = canvas[cy - reach : cy + reach + 1, cx - reach : cx + reach + 1]
        px, py, score = _peak(cv2.matchTemplate(area, template, cv2.TM_CCOEFF_NORMED))
        found = np.array([[[cx - window + px, cy - window + py]]], np.float64)
        corners.append(tuple(cv2.perspectiveTransform(found, to_photo).ravel().tolist()))
        scores.append(score)

    return Marks(corners, scores)


def find_marks(photo: np.ndarray, profile: TargetProfile) -> Marks | None:
    """The four marks in photo pixels, or ``None`` if the black disc was not found."""
    if mm_per_px(profile) is None:
        return None

    grey = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY) if photo.ndim == 3 else photo
    to_canon = _find_disc(grey, profile)

    if to_canon is None:
        return None

    dst = np.asarray(profile.manual_corners_canon, np.float32)
    # The circle finds the mark among everything else near a corner; the cross alone then
    # places it, since its centre is the point and does not move with a scale error.
    circle = _template(profile, circle=True)
    cross = _template(profile, circle=False)
    marks = _pass(grey, to_canon, profile, circle, COARSE_WINDOW)

    for _ in range(2):
        to_canon = cv2.getPerspectiveTransform(np.asarray(marks.corners, np.float32), dst)
        marks = _pass(grey, to_canon, profile, cross, FINE_WINDOW)

    # Scored by the circle, and only once perspective is gone: a bare cross also matches
    # line crossings and digits, and a circle seen through the disc's rough frame is an
    # ellipse that matches nothing well.
    to_canon = cv2.getPerspectiveTransform(np.asarray(marks.corners, np.float32), dst)

    return Marks(marks.corners, _pass(grey, to_canon, profile, circle, FINE_WINDOW).scores)
