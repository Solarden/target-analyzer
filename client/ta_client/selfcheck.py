"""Assert the client's half of the canonical-frame contract, without a camera or a window.

Framework-free on purpose, so it runs on a box with no pytest:
``python -m ta_client.selfcheck``. Nothing here opens a highgui window, and nothing is
random — a self-check that flakes is worse than no self-check.

The registration case is a real photo run backwards: the board is rendered, warped by a
*known* homography into a synthetic photograph, and then handed to the same
:func:`ta_client.register.register` the Mac runs. Recovering the known geometry from it
is the property the whole product rests on.
"""

import hashlib
import re
import tempfile
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from ta_client.board import render_markers, render_svg
from ta_client.load import load_stripped
from ta_client.package import build_payload
from ta_client.pick import _loupe, _Picking
from ta_client.register import Registration, TooFewMarkers, refine_to_rings, register, warp
from ta_shared.payload import Hit, SessionMeta, ShipPayload
from ta_shared.profile import TargetProfile, load_profile

PROFILES = Path(__file__).resolve().parents[2] / "profiles"
PROFILE_JSON = PROFILES / "issf_precision.json"
# The profile with a black aiming area, which is what refine_to_rings measures.
PISTOL_JSON = PROFILES / "issf_pistol_50m.json"
# ArUco cannot read a marker with no quiet zone, and the board's sit flush to the edge.
# This stands in for the paper margin a printed sheet must therefore have.
PAD = 100
PHOTO_SIZE = (1800, 1400)
# Mild on purpose: the question is whether the geometry is recovered, not how far the
# detector bends before it breaks.
PHOTO_QUAD = np.float32([[320, 210], [1520, 160], [1610, 1180], [250, 1240]])
PROBES = np.float32(
    [[500, 500], [0, 0], [1000, 0], [1000, 1000], [0, 1000], [500, 0], [1000, 500]]
).reshape(-1, 1, 2)
TOLERANCE_PX = 2.0


def _synthetic_photo(profile: TargetProfile) -> tuple[np.ndarray, np.ndarray]:
    """A fake photograph of the board, plus the true photo-to-canonical homography."""
    board = render_markers(profile)
    canvas = cv2.copyMakeBorder(board, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(255,) * 3)
    height, width = canvas.shape[:2]
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    to_photo = cv2.getPerspectiveTransform(corners, PHOTO_QUAD)
    photo = cv2.warpPerspective(canvas, to_photo, PHOTO_SIZE, borderValue=(255,) * 3)
    un_pad = np.array([[1, 0, -PAD], [0, 1, -PAD], [0, 0, 1]], float)

    return photo, un_pad @ np.linalg.inv(to_photo)


def _hide_markers(
    photo: np.ndarray, profile: TargetProfile, truth: np.ndarray, marker_ids: list[int]
) -> np.ndarray:
    """Paint over the given markers, as a torn or shadowed corner would."""
    out = photo.copy()

    for marker_id in marker_ids:
        canonical = np.float32(profile.board.markers[marker_id]).reshape(-1, 1, 2)
        in_photo = cv2.perspectiveTransform(canonical, np.linalg.inv(truth))
        cv2.fillConvexPoly(out, in_photo.reshape(-1, 2).astype(np.int32), (255, 255, 255))

    return out


def _jpeg_with_exif() -> bytes:
    """A JPEG carrying an EXIF segment, since OpenCV can read metadata but never write it."""
    _ok, buffer = cv2.imencode(".jpg", np.full((64, 64, 3), 200, np.uint8))
    plain = buffer.tobytes()
    # A valid, empty little-endian TIFF header: byte order, magic, IFD at offset 8, no entries.
    app1 = b"Exif\x00\x00" + b"II*\x00\x08\x00\x00\x00\x00\x00"

    return plain[:2] + b"\xff\xe1" + (len(app1) + 2).to_bytes(2, "big") + app1 + plain[2:]


def selfcheck() -> None:
    profile = load_profile(PROFILE_JSON)
    photo, truth = _synthetic_photo(profile)

    # --- registration recovers the known geometry ---
    registration = register(photo, profile)
    assert registration.source == "aruco"
    assert registration.n_markers == 4, registration.n_markers
    assert registration.n_inliers == 16, registration.n_inliers
    assert registration.residual < 1.0, registration.residual
    assert registration.normalized.shape == (profile.canon_size_px, profile.canon_size_px, 3)

    # Homographies are equal only up to scale, so an element-wise comparison fails even on
    # a perfect fit. Compare what they do to points instead.
    in_photo = cv2.perspectiveTransform(PROBES, np.linalg.inv(truth))
    recovered = cv2.perspectiveTransform(in_photo, registration.homography)
    drift = np.linalg.norm(recovered.reshape(-1, 2) - PROBES.reshape(-1, 2), axis=1)
    assert drift.max() < TOLERANCE_PX, drift.max()

    round_trip = cv2.perspectiveTransform(recovered, np.linalg.inv(registration.homography))
    assert np.allclose(round_trip, in_photo, atol=1e-6)

    # --- the board degrades to the minimum, then refuses ---
    assert register(_hide_markers(photo, profile, truth, [3]), profile).n_markers == 3

    try:
        register(_hide_markers(photo, profile, truth, [2, 3]), profile)
        raise AssertionError("two markers must not be enough to register")
    except TooFewMarkers:
        pass

    # --- the identity hash is of stripped bytes ---
    with tempfile.TemporaryDirectory() as tmp:
        dirty = Path(tmp) / "dirty.jpg"
        dirty.write_bytes(_jpeg_with_exif())
        assert b"Exif\x00\x00" in dirty.read_bytes()

        _bgr, stripped, digest = load_stripped(dirty)
        assert b"Exif\x00\x00" not in stripped  # §12.4 — no camera metadata reaches the wire
        assert digest == hashlib.sha256(stripped).hexdigest()
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        # Deterministic, which is what an idempotency key needs. Not idempotent: JPEG is
        # lossy, so re-encoding the stripped bytes again would give different ones.
        assert load_stripped(dirty)[1] == stripped

    # --- the payload survives the wire ---
    payload = build_payload(
        image_sha256=digest,
        profile=profile,
        registration=registration,
        hits=[Hit(x_canon=500, y_canon=500), Hit(x_canon=510, y_canon=520)],
        session=SessionMeta(
            gun="CZ 75",
            distance_m=25,
            shot_at=date(2026, 9, 13),
            target_profile=profile.name,
            target_profile_version=profile.version,
        ),
    )
    # A numpy scalar in the bare params dict survives in memory and only explodes on the
    # wire, long after the clicking is done.
    assert ShipPayload.model_validate_json(payload.model_dump_json()) == payload
    assert payload.canon_size_px == profile.canon_size_px
    assert (payload.method, payload.model) == ("manual", None)
    assert len(payload.params["homography"]) == 3

    # --- the printed rings settle a frame that came in stretched ---
    pistol = load_profile(PISTOL_JSON)
    middle = pistol.canon_size_px // 2
    flat = np.full((pistol.canon_size_px, pistol.canon_size_px, 3), 255, np.uint8)
    cv2.circle(flat, (middle, middle), pistol.ring_radius_px(pistol.black_ring), (20, 20, 20), -1)

    # A tape measure 5 % narrow and 4 % tall, which reaches the frame as a per-axis stretch.
    stretched = np.array([[1.05, 0.0, -30.0], [0.0, 0.96, 25.0], [0.0, 0.0, 1.0]])
    skewed = Registration(
        homography=stretched,
        normalized=warp(flat, stretched, pistol.canon_size_px),
        residual=None,
        n_markers=0,
        n_inliers=4,
        source="manual",
    )
    settled = refine_to_rings(flat, skewed, pistol)
    assert settled.ring_correction is not None, "the black edge should have been readable"
    assert abs(settled.ring_correction[0] - 1 / 1.05) < 0.01, settled.ring_correction
    assert abs(settled.ring_correction[1] - 1 / 0.96) < 0.01, settled.ring_correction

    # Undone, the frame is where it started, so the corrected homography is the identity.
    corners = np.float32(
        [[0, 0], [pistol.canon_size_px, 0], [middle, middle], [0, pistol.canon_size_px]]
    ).reshape(-1, 1, 2)
    settled_back = cv2.perspectiveTransform(corners, settled.homography)
    assert np.abs(settled_back - corners).max() < 3.0, np.abs(settled_back - corners).max()

    # A profile with no black has nothing to measure, and must say so rather than guess.
    assert refine_to_rings(photo, registration, profile).ring_correction is None

    # The round trip above runs on a registration the rings never touched; this is the
    # shape that actually ships, and it carries one field more.
    settled_payload = build_payload(
        image_sha256=digest,
        profile=pistol,
        registration=settled,
        hits=[Hit(x_canon=750, y_canon=750)],
        session=SessionMeta(
            gun="Glock",
            distance_m=15,
            shot_at=date(2026, 9, 13),
            target_profile=pistol.name,
            target_profile_version=pistol.version,
        ),
    )
    assert ShipPayload.model_validate_json(settled_payload.model_dump_json()) == settled_payload

    # --- the picker turns screen clicks into full-resolution coordinates ---
    # Driven directly, without a window: every defect here is a silently wrong coordinate.
    picking = _Picking(scale=0.5, max_points=2)
    picking.on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 60, 0)
    assert picking.points == [(200.0, 120.0)], picking.points  # divided back by the scale

    picking.on_mouse(cv2.EVENT_MOUSEMOVE, 120, 60, cv2.EVENT_FLAG_LBUTTON)
    assert picking.points == [(240.0, 120.0)], "a drag with the button held moves the point"

    # Released off-window, so no LBUTTONUP arrived: moving on must not keep dragging, or
    # the point follows the cursor for the rest of the session.
    picking.on_mouse(cv2.EVENT_MOUSEMOVE, 400, 400, 0)
    assert picking.points == [(240.0, 120.0)], picking.points
    assert picking.dragging is None

    picking.on_mouse(cv2.EVENT_LBUTTONDOWN, 300, 300, 0)
    picking.on_mouse(cv2.EVENT_LBUTTONUP, 300, 300, 0)
    picking.on_mouse(cv2.EVENT_LBUTTONDOWN, 500, 500, 0)
    assert len(picking.points) == 2, "max_points must cap what a click can add"

    # Undo mid-drag: the dragged index may be the one just dropped.
    picking.on_mouse(cv2.EVENT_LBUTTONDOWN, 150, 150, 0)
    picking.on_mouse(cv2.EVENT_RBUTTONDOWN, 150, 150, 0)
    picking.on_mouse(cv2.EVENT_MOUSEMOVE, 160, 160, cv2.EVENT_FLAG_LBUTTON)
    assert len(picking.points) == 1, picking.points

    # The loupe crops around the cursor, so a mark at the very edge of the frame is the
    # case that would slice out of bounds.
    big = np.full((400, 600, 3), 120, np.uint8)
    view = np.full((360, 540, 3), 120, np.uint8)

    for at in ((0.0, 0.0), (300.0, 200.0), (599.0, 399.0)):
        before = view.copy()
        _loupe(big, view, at, 0.5)
        assert not np.array_equal(view, before), f"the loupe drew nothing at {at}"

    # Too small to hold the loupe: skipped, never a broadcast error mid-redraw.
    cramped = np.full((100, 100, 3), 120, np.uint8)
    _loupe(big, cramped, (300.0, 200.0), 0.5)
    assert cramped.min() == cramped.max() == 120

    # --- the printable sheet is to scale ---
    svg = render_svg(profile)
    assert 'width="175.500mm"' in svg  # 155.5 mm of target plus the two 10 mm margins
    assert svg.count("<circle") == profile.n_rings
    assert svg.count("<g transform") == len(profile.board.markers) + 1  # +1 for the margin shift

    print("client self-check OK")


if __name__ == "__main__":
    selfcheck()
