"""The one interactive surface: an OpenCV window for clicking points on an image.
See implementation.md §9.

Every highgui call in the client lives here, and so does every macOS quirk that comes
with it. Two callers — the four-corner registration fallback and the hit picker — get
identical undo and adjust behaviour because there is only one picker.

Points are held in **full-resolution image coordinates** and scaled only for drawing.
A large photo has to be shrunk to fit a screen, and a click that is not divided back by
that factor produces coordinates that look plausible and are wrong.
"""

import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

# cv2 cannot ask the OS how big the screen is, so this is a fixed ceiling.
# ponytail: a larger display just gets a smaller window than it could have.
MAX_DISPLAY_PX = 1400
GRAB_PX = 12  # click within this (on screen) to drag an existing point instead of adding one
# A shrunk photo hides the detail a corner has to land on, so the pixels under the
# cursor are magnified into a corner of the window.
LOUPE_PX = 200
# Measured in *displayed* pixels, so the magnification does not change with photo size.
LOUPE_WINDOW_PX = 44
POINT_COLOUR = (0, 140, 255)
READY_COLOUR = (0, 190, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
KEYS = "left-click add · drag to move · u / right-click undo · Enter confirm · Esc cancel"
ACCEPT_KEYS = (13, 10, ord("y"))  # Cocoa sends 13 for Return where other backends send 10


def _fit(image: np.ndarray) -> tuple[np.ndarray, float]:
    scale = min(1.0, MAX_DISPLAY_PX / max(image.shape[:2]))

    if scale == 1.0:
        return image, scale

    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def _open(title: str, image: np.ndarray) -> None:
    # AUTOSIZE keeps one window unit equal to one displayed pixel, which is what makes the
    # single scale factor above enough to map a click back to the image.
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    # Without this the window opens behind the terminal and the run looks hung.
    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(title, image)
    cv2.waitKey(1)


def _close(title: str) -> None:
    cv2.destroyWindow(title)

    # macOS only lets go of the window once its event loop has been pumped a few times.
    for _ in range(4):
        cv2.waitKey(1)


def _banner(image: np.ndarray, lines: list[str]) -> None:
    """Draw text on a dimmed strip, so it reads over white paper and a black bull alike.

    A darkened panel rather than outlined glyphs: at the size this text is drawn, an
    outline thick enough to separate it from the paper fills the letters in.
    """
    if not lines:
        return

    scale = max(0.55, image.shape[1] / 1800)
    thickness = 1 if scale < 0.9 else 2
    sizes = [cv2.getTextSize(text, FONT, scale, thickness)[0] for text in lines]
    step = max(height for _w, height in sizes) + 14
    panel = image[: step * len(lines) + 12, : min(image.shape[1], max(w for w, _h in sizes) + 28)]
    cv2.addWeighted(panel, 0.25, np.zeros_like(panel), 0.75, 0.0, panel)

    for row, text in enumerate(lines):
        origin = (14, 12 + step * row + sizes[row][1])
        cv2.putText(image, text, origin, FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _loupe(image: np.ndarray, view: np.ndarray, at: tuple[float, float], scale: float) -> None:
    """Blit a magnified crop of the full-resolution image into a corner of the view."""
    view_h, view_w = view.shape[:2]

    if view_h < LOUPE_PX + 24 or view_w < LOUPE_PX + 24:
        return  # nothing to magnify into

    span = max(8, round(LOUPE_WINDOW_PX / scale))
    height, width = image.shape[:2]
    left = int(round(at[0])) - span // 2
    top = int(round(at[1])) - span // 2
    crop = np.zeros((span, span, 3), image.dtype)
    # Clamped and offset rather than skipped, so the loupe still works on a corner mark
    # sitting at the very edge of the frame.
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + span), min(height, top + span)

    if x1 <= x0 or y1 <= y0:
        return

    crop[y0 - top : y1 - top, x0 - left : x1 - left] = image[y0:y1, x0:x1]
    patch = cv2.resize(crop, (LOUPE_PX, LOUPE_PX), interpolation=cv2.INTER_LINEAR)
    cv2.drawMarker(patch, (LOUPE_PX // 2, LOUPE_PX // 2), POINT_COLOUR, cv2.MARKER_CROSS, 22, 1)
    cv2.rectangle(patch, (0, 0), (LOUPE_PX - 1, LOUPE_PX - 1), (40, 40, 40), 1)

    # Bottom corner away from the cursor, so it never covers what is being aimed at.
    side = 12 if at[0] * scale > view_w / 2 else view_w - LOUPE_PX - 12
    view[view_h - LOUPE_PX - 12 : view_h - 12, side : side + LOUPE_PX] = patch


def _nearest(
    points: list[tuple[float, float]], target: tuple[float, float], radius: float
) -> int | None:
    for index, (x, y) in enumerate(points):
        if abs(x - target[0]) <= radius and abs(y - target[1]) <= radius:
            return index

    return None


@dataclass
class _Picking:
    """The picker's state, out of the callback closure so it can be driven without a window.

    Clicks arrive as screen coordinates and are stored at full resolution; every bug this
    class can have is a silently wrong hit coordinate, which is why it is testable.
    """

    scale: float
    max_points: int | None = None
    points: list[tuple[float, float]] = field(default_factory=list)
    dragging: int | None = None
    hover: tuple[float, float] | None = None

    def undo(self) -> None:
        """Drop the last point, and end any drag: its index may be the one just dropped."""
        if self.points:
            self.points.pop()

        self.dragging = None

    def on_mouse(self, event: int, x: int, y: int, flags: int, _param: object = None) -> None:
        full = (x / self.scale, y / self.scale)
        self.hover = full

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = _nearest(self.points, full, GRAB_PX / self.scale)

            if self.dragging is None and (
                self.max_points is None or len(self.points) < self.max_points
            ):
                self.points.append(full)
                self.dragging = len(self.points) - 1
        elif event == cv2.EVENT_MOUSEMOVE:
            # The button flag, not merely the last event seen: a release outside the window
            # delivers no LBUTTONUP, and the point would then follow the cursor for good.
            if self.dragging is not None and flags & cv2.EVENT_FLAG_LBUTTON:
                self.points[self.dragging] = full
            else:
                self.dragging = None
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo()


def pick_points(
    image: np.ndarray,
    title: str,
    *,
    exact: int | None = None,
    max_points: int | None = None,
) -> list[tuple[float, float]] | None:
    """Collect clicked points, or ``None`` if the user cancelled.

    ``None`` and ``[]`` mean different things: an empty list is a real answer (a string
    where every shot missed the paper), and collapsing the two would ship an empty
    session as though it had been confirmed.
    """
    canvas, scale = _fit(image)
    state = _Picking(scale=scale, max_points=max_points)
    points = state.points

    # Also on stderr because waitKey only sees keys while the cv2 window has focus, and a
    # user who has clicked back to the terminal needs to know why nothing responds.
    print(f"{title}: {KEYS}", file=sys.stderr)
    _open(title, canvas)
    cv2.setMouseCallback(title, state.on_mouse)

    try:
        while True:
            ready = exact is None or len(points) == exact
            count = f"{len(points)}/{exact}" if exact else str(len(points))
            view = canvas.copy()

            for index, (x, y) in enumerate(points, start=1):
                centre = (round(x * scale), round(y * scale))
                colour = READY_COLOUR if ready else POINT_COLOUR
                cv2.circle(view, centre, 7, colour, 1, cv2.LINE_AA)
                cv2.drawMarker(view, centre, colour, cv2.MARKER_CROSS, 9, 1)
                cv2.putText(
                    view,
                    str(index),
                    (centre[0] + 9, centre[1] - 9),
                    FONT,
                    0.4,
                    colour,
                    1,
                    cv2.LINE_AA,
                )

            status = f"{count} points — Enter to confirm" if ready else f"{count} points"
            # In the window as well as the title bar, which macOS truncates.
            _banner(view, [title, status, KEYS])

            if state.hover is not None:
                _loupe(image, view, state.hover, scale)

            cv2.imshow(title, view)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                return None

            if key in ACCEPT_KEYS and ready:
                return list(points)

            if key in (ord("u"), ord("U")):
                state.undo()
    finally:
        _close(title)


def confirm(image: np.ndarray, title: str, lines: list[str]) -> bool:
    """Show an image and wait for Enter (accepted) or any other key (not accepted)."""
    canvas, _scale = _fit(image)
    view = canvas.copy()
    _banner(view, lines)

    for line in lines:
        print(line, file=sys.stderr)

    _open(title, view)

    try:
        while True:
            cv2.imshow(title, view)
            key = cv2.waitKey(20) & 0xFF

            if key in ACCEPT_KEYS:
                return True

            if key != 255:  # 255 is "no key was pressed"
                return False
    finally:
        _close(title)
