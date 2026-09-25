"""Assert the client's half of the canonical-frame contract, without a camera or a window.

Framework-free on purpose, so it runs on a box with no pytest:
``python -m ta_client.selfcheck``. Nothing here opens a highgui window, and nothing is
random — a self-check that flakes is worse than no self-check.

The registration case is a real photo run backwards: the board is rendered, warped by a
*known* homography into a synthetic photograph, and then handed to the same
:func:`ta_client.register.register` the Mac runs. Recovering the known geometry from it
is the property the whole product rests on.
"""

import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterator
from datetime import date
from math import hypot
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import httpx2
import numpy as np

from ta_client import cv_blob, cv_blob_vlm, marks, pick, ship, vlm, yolo, yolo_data
from ta_client.board import render_markers, render_svg
from ta_client.config import Settings
from ta_client.detector import DetectorError
from ta_client.load import load_stripped
from ta_client.package import build_payload
from ta_client.pick import _loupe, _Picking
from ta_client.register import (
    Registration,
    TooFewMarkers,
    _ring_lines,
    refine_to_rings,
    register,
    register_interactive,
    warp,
)
from ta_client.ship import _parts
from ta_shared.agreement import MATCH_TOL_MM, agreement
from ta_shared.payload import Hit, SessionMeta, ShipPayload
from ta_shared.profile import TargetProfile, hole_diam_px, load_profile, mm_per_px

# Stands in for the machine token wherever a Settings is built here. Not a credential:
# nothing in this file reaches a network, so no server ever sees it.
FAKE_TOKEN = "tok"  # nosec B105

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
# The corner marks overhang the canonical square, so the sheet carrying them is wider.
MARK_PAD = 200
# Keystoned hard enough that the disc's frame alone misses the marks by tens of pixels, on a
# dark background surrounding the whole sheet: that nests the black disc inside another
# contour, where a search of outer contours never looks.
MARKS_PHOTO_SIZE = (2400, 2800)
MARKS_QUAD = np.float32([[420, 300], [2100, 120], [2300, 2650], [150, 2500]])


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


def _queue(outbox: Path, *names: str) -> None:
    """Put folders in the outbox without going through enqueue, which names them itself."""
    for name in names:
        folder = outbox / name
        folder.mkdir(parents=True)
        (folder / "payload.json").write_text("{}", encoding="utf-8")
        (folder / "normalized.png").write_bytes(b"png")


def _clean_target(profile: TargetProfile) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """An unmarked target with holes punched at known places, half of them on the black.

    Polarity is the point: paper is light and a hole through it is dark, while on the
    black aiming area the torn edge catches the light and the same hole is pale.
    """
    size = profile.canon_size_px
    centre = (size // 2, size // 2)
    image = np.full((size, size, 3), 235, np.uint8)
    cv2.circle(image, centre, profile.ring_radius_px(profile.black_ring), (30, 30, 30), -1)

    for radius in profile.ring_radii_px:
        cv2.circle(image, centre, radius, (90, 90, 90), 1, cv2.LINE_AA)

    radius = round(hole_diam_px(profile) / 2)
    black = profile.ring_radius_px(profile.black_ring)
    holes = [
        (centre[0] + black * 0.4, centre[1] - black * 0.3),
        (centre[0] - black * 0.5, centre[1] + black * 0.2),
        (centre[0] + black * 2.0, centre[1] + black * 0.4),
        (centre[0] - black * 1.6, centre[1] - black * 1.5),
    ]

    for x, y in holes:
        on_black = np.hypot(x - centre[0], y - centre[1]) <= black
        colour = (205, 205, 205) if on_black else (15, 15, 15)
        cv2.circle(image, (round(x), round(y)), radius, colour, -1, cv2.LINE_AA)

    return image, holes


def _settings(**overrides) -> Settings:
    """Settings from these values and the class defaults — never a personal config."""
    return Settings(_env_file=None, **overrides)


def _reply(status_code: int) -> SimpleNamespace:
    """The two attributes flush reads off a response."""
    return SimpleNamespace(status_code=status_code, text="detail")


def _jpeg_with_exif() -> bytes:
    """A JPEG carrying an EXIF segment, since OpenCV can read metadata but never write it."""
    _ok, buffer = cv2.imencode(".jpg", np.full((64, 64, 3), 200, np.uint8))
    plain = buffer.tobytes()
    # A valid, empty little-endian TIFF header: byte order, magic, IFD at offset 8, no entries.
    app1 = b"Exif\x00\x00" + b"II*\x00\x08\x00\x00\x00\x00\x00"

    return plain[:2] + b"\xff\xe1" + (len(app1) + 2).to_bytes(2, "big") + app1 + plain[2:]


def _marked_photo(profile: TargetProfile, *, with_marks: bool) -> tuple[np.ndarray, np.ndarray]:
    """A photo of a sheet with no marker board, and where its four corner marks landed."""
    sheet, _holes = _clean_target(profile)
    sheet = cv2.copyMakeBorder(sheet, *[MARK_PAD] * 4, cv2.BORDER_CONSTANT, value=(235,) * 3)
    corners = np.float32(profile.manual_corners_canon) + MARK_PAD
    per_mm = 1 / mm_per_px(profile)
    half = round(marks.CROSS_HALF_MM * per_mm)

    for x, y in corners.round().astype(int).tolist() if with_marks else []:
        cv2.circle(sheet, (x, y), round(marks.MARK_RADIUS_MM * per_mm), (40,) * 3, 3, cv2.LINE_AA)
        cv2.line(sheet, (x - half, y), (x + half, y), (40,) * 3, 3, cv2.LINE_AA)
        cv2.line(sheet, (x, y - half), (x, y + half), (40,) * 3, 3, cv2.LINE_AA)

    side = sheet.shape[0]
    square = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    to_photo = cv2.getPerspectiveTransform(square, MARKS_QUAD)
    photo = cv2.warpPerspective(sheet, to_photo, MARKS_PHOTO_SIZE, borderValue=(60,) * 3)

    return photo, cv2.perspectiveTransform(corners.reshape(-1, 1, 2), to_photo).reshape(-1, 2)


def _registering(
    photo: np.ndarray,
    profile: TargetProfile,
    *,
    confirms: list[bool],
    clicks: list[list[float]] | None = None,
) -> list[tuple[str, object]]:
    """Run register_interactive with the windows replaced; what it asked for, in order."""
    seen: list[tuple[str, object]] = []
    answers: Iterator[bool] = iter(confirms)

    def _confirm(_image: np.ndarray, _title: str, lines: list[str]) -> bool:
        seen.append(("confirm", lines[1]))

        return next(answers)

    def _pick(_image: np.ndarray, _title: str, **kwargs) -> list[tuple[float, float]]:
        seen.append(("pick", kwargs.get("initial")))
        assert clicks is not None, f"the corner picker opened where it should not have: {seen}"

        return [tuple(point) for point in clicks]

    with (
        mock.patch.object(pick, "confirm", _confirm),
        mock.patch.object(pick, "pick_points", _pick),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        assert register_interactive(photo, profile) is not None

    return seen


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

    # A black with no ring lines round it is a scale, not a set of lines to bend onto.
    assert settled.radial is None, settled.radial

    # --- then each printed line is bent onto its ring, however unevenly the print drifts ---
    outer = pistol.ring_radii_px[-1]
    black = pistol.ring_radius_px(pistol.black_ring)
    drifting = np.full_like(flat, 255)
    # Drifting outward from nothing at the centre, the way real sheets do.
    shift = 4

    def _drawn(radius: float) -> int:
        return round(radius * (1 + 0.025 * radius / outer) * (1 << shift))

    centre_fp = (middle << shift, middle << shift)
    cv2.circle(drifting, centre_fp, _drawn(black), (20,) * 3, -1, cv2.LINE_AA, shift)

    for ring in range(1, pistol.n_rings + 1):
        radius = pistol.ring_radius_px(ring)
        colour = (20,) * 3 if radius > black else (235,) * 3

        if ring != pistol.black_ring:
            cv2.circle(drifting, centre_fp, _drawn(radius), colour, 2, cv2.LINE_AA, shift)

    as_shot = Registration(np.eye(3), drifting, None, 0, 4, "manual")
    bent = refine_to_rings(drifting, as_shot, pistol)
    assert bent.radial is not None, "every ring line was printed and should have been read"
    read = _ring_lines(bent.normalized, pistol)
    off = {ring: read[ring] - pistol.ring_radius_px(ring) for ring in read}
    assert len(off) == pistol.n_rings - 1 and max(map(abs, off.values())) < 0.5, off

    # A profile with no black has nothing to measure, and must say so rather than guess.
    assert refine_to_rings(photo, registration, profile).ring_correction is None

    # --- the printed corner marks register a sheet with no marker board ---
    marked, where = _marked_photo(pistol, with_marks=True)
    found = marks.find_marks(marked, pistol)
    assert found is not None and found.confident, found
    miss = np.linalg.norm(np.asarray(found.corners) - where, axis=1)
    assert miss.max() < 0.25, miss

    # The disc alone is not a registration: without the marks there is nothing to vouch.
    bare = marks.find_marks(_marked_photo(pistol, with_marks=False)[0], pistol)
    assert bare is not None and not bare.confident, bare.scores

    # --- and the interactive path routes between them without a window ---
    # Clear marks go straight to the confirm; turned down, the picker opens holding them.
    seen = _registering(marked, pistol, confirms=[True])
    assert [step for step, _ in seen] == ["confirm"], seen
    assert "printed corner marks" in seen[0][1], seen

    seen = _registering(marked, pistol, confirms=[False, True], clicks=where.tolist())
    assert [step for step, _ in seen] == ["confirm", "pick", "confirm"], seen
    held = seen[1][1]
    assert held is not None and np.allclose(held, found.corners), "the picker should hold the marks"

    # With no mark found there is nothing worth seeding, so the picker opens empty.
    bare_photo = _marked_photo(pistol, with_marks=False)[0]
    seen = _registering(bare_photo, pistol, confirms=[True], clicks=where.tolist())
    assert seen[0] == ("pick", None), seen

    # The round trip above runs on a registration the rings never touched; this is the
    # shape that actually ships, and it carries two fields more.
    settled_payload = build_payload(
        image_sha256=digest,
        profile=pistol,
        registration=bent,
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

    seeded = _Picking(scale=1.0, points=[(10.0, 10.0), (50.0, 50.0), (90.0, 90.0)])
    seeded.hover = (50.0, 50.0)
    seeded.undo()
    assert seeded.points == [(10.0, 10.0), (90.0, 90.0)], "the point under the cursor goes"

    seeded.hover = (400.0, 400.0)
    seeded.undo()
    assert seeded.points == [(10.0, 10.0)], "with nothing under the cursor, the last one goes"

    seeded.clear()
    assert seeded.points == []

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

    # --- the blob detector finds holes on clean paper, whichever way they read ---
    pistol = load_profile(PISTOL_JSON)
    target, punched = _clean_target(pistol)
    proposal = cv_blob.detect(target, pistol)
    found = agreement(
        punched,
        [(hit.x_canon, hit.y_canon) for hit in proposal],
        MATCH_TOL_MM / mm_per_px(pistol),
    )
    assert found.matched == len(punched), (found, len(proposal))
    assert found.spurious == 0, found
    assert all(0 < hit.confidence <= 1 for hit in proposal), proposal
    # Strongest first, which is the order a person reads a proposal in.
    assert [hit.confidence for hit in proposal] == sorted(
        (hit.confidence for hit in proposal), reverse=True
    )

    # Neither of these can be inferred from the pixels, and a guess at either mis-tunes
    # every step of the detector in silence.
    try:
        cv_blob.detect(target, pistol.model_copy(update={"target_diam_mm": None}))
        raise AssertionError("a profile with no scale must be refused")
    except DetectorError as exc:
        assert "target_diam_mm" in str(exc), str(exc)

    try:
        cv_blob.detect(target, load_profile(PROFILE_JSON))
        raise AssertionError("an image that is not the canonical square must be refused")
    except ValueError as exc:
        assert not isinstance(exc, DetectorError), "a wrong frame is a bug, not a bad day"
        assert "canonical square" in str(exc), str(exc)

    # --- the vision detector, without a vision model ---
    # Everything except the model: the prompt that goes out, the answer that comes back,
    # and what each end refuses. The sender is injected, so nothing here reaches a network.
    asked = _settings(vlm_base_url="https://model.invalid/v1", vlm_model="test-model")
    sent: list[httpx2.Request] = []

    def answering(content: str, status_code: int = 200):
        def send(_client, request):
            sent.append(request)

            return SimpleNamespace(
                status_code=status_code,
                json=lambda: {"choices": [{"message": {"content": content}}]},
                text=content,
            )

        return send

    # Fenced, prefaced, a brace in the prose, and one hole outside the frame — every way a
    # real answer arrives wrapped in something.
    holes = '{"holes": [{"x": 700, "y": 800}, {"x": -5, "y": 10}]}'
    fenced = f"Sure! (see {{the rings}}):\n```json\n{holes}\n```"
    proposal = vlm.detect(target, pistol, settings=asked, send=answering(fenced))

    assert [(h.x_canon, h.y_canon) for h in proposal] == [(700.0, 800.0)], proposal
    assert vlm.parse_hits('{"holes": []}', 1500) == ([], 0), "an empty answer is a real answer"
    assert vlm.model_name(asked) == "test-model", vlm.model_name(asked)
    assert all(hit.confidence is None for hit in proposal), "no confidence is invented"

    body = json.loads(sent[0].read())
    content = body["messages"][0]["content"]

    assert body["model"] == "test-model"
    assert body["temperature"] == 0.0, "a detector that answers differently each run"
    assert str(pistol.canon_size_px) in content[0]["text"], "the frame size is in the prompt"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "authorization" not in {k.lower() for k in sent[0].headers}, "no key, no header"

    def parts(_client, request):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": [{"text": "{}"}]}}]},
            text="[]",
        )

    for send, hint in (
        (parts, "is list, not text"),
        (answering("there is no object here"), "no JSON object"),
        (answering('{"oops": []}'), "holes"),
        (answering('{"holes": {"1": {"x": 1, "y": 2}}}'), "not a list"),
        (answering('{"holes": [{"x": 1}]}'), "is not a hole"),
        (answering("{}", status_code=503), "503"),
    ):
        try:
            vlm.detect(target, pistol, settings=asked, send=send)
            raise AssertionError(f"{hint} must be refused")
        except vlm.VlmError as exc:
            assert hint in str(exc), str(exc)

    # Config refusals, every one of them survivable: the run drops to marking by hand.
    for settings, hint in (
        (_settings(), "TA_VLM_BASE_URL"),
        (_settings(vlm_base_url="model.invalid/v1", vlm_model="m"), "full http:// or https://"),
        (
            _settings(vlm_base_url="http://model.invalid/v1", vlm_model="m", vlm_api_key="k"),
            "cross the network in the clear",
        ),
    ):
        try:
            vlm.detect(target, pistol, settings=settings, send=answering("{}"))
            raise AssertionError(f"{hint} must be refused")
        except DetectorError as exc:
            assert hint in str(exc), str(exc)

    vlm.check(
        _settings(vlm_base_url="http://127.0.0.1:1/v1", vlm_model="m", vlm_api_key="k"), pistol
    )

    try:
        vlm.detect(np.zeros((10, 10, 3), np.uint8), pistol, settings=asked, send=answering("{}"))
        raise AssertionError("an image that is not the canonical square must be refused")
    except ValueError as exc:
        assert not isinstance(exc, DetectorError), "a wrong frame is a bug, not a bad day"
        assert "canonical square" in str(exc), str(exc)

    # --- the blob filter the model vets, without a vision model ---
    candidates = cv_blob.detect(target, pistol)
    crops: list[dict] = []

    def judging(*verdicts: str):
        answers = iter(verdicts)

        def send(_client, request):
            crops.append(json.loads(request.read()))
            content = json.dumps({"answer": next(answers)})

            return SimpleNamespace(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": content}}]},
                text="",
            )

        return send

    assert candidates, "the synthetic target must give the blob filter something to propose"

    verdicts = ["hole" if index % 2 == 0 else "not" for index in range(len(candidates))]
    kept = cv_blob_vlm.detect(target, pistol, settings=asked, send=judging(*verdicts))
    expected = [c for c, verdict in zip(candidates, verdicts, strict=True) if verdict == "hole"]

    assert len(crops) == len(candidates), "one question per candidate"
    assert [(h.x_canon, h.y_canon) for h in kept] == [(c.x_canon, c.y_canon) for c in expected], (
        "the model replaces the judgement, never the coordinates"
    )

    sent_crop = crops[0]["messages"][0]["content"]

    assert sent_crop[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "centre" in sent_crop[0]["text"], "the prompt has to say which object it means"

    for answer, hint in (
        ('{"answer": "maybe"}', "neither 'hole' nor 'not'"),
        ('{"nope": 1}', "'answer' key"),
    ):
        try:
            cv_blob_vlm.verdict(answer)
            raise AssertionError(f"{hint} must be refused")
        except vlm.VlmError as exc:
            assert hint in str(exc), str(exc)

    try:
        cv_blob_vlm.detect(target, pistol, settings=_settings(), send=judging("hole"))
        raise AssertionError("a missing endpoint must be refused")
    except DetectorError as exc:
        assert "TA_VLM_BASE_URL" in str(exc), str(exc)

    hedged = cv_blob_vlm.detect(
        target, pistol, settings=asked, send=answering('{"answer": "uncertain"}')
    )

    assert len(hedged) == len(candidates), "an unreadable verdict keeps its candidate"

    try:
        cv_blob_vlm.crop(target, 10, 10, 50, pistol.canon_size_px)
        raise AssertionError("an unpadded frame must be refused")
    except ValueError as exc:
        assert "padded" in str(exc), str(exc)

    # --- the trained detector, without weights and without torch ---
    # Everything except the model: which tiles are cut, where a box inside one lands, and
    # what two tiles reporting one hole do. The predictor is injected, so nothing loads.
    tile, stride, margin = yolo.geometry(pistol)
    # The seam rule holds only because the safe interiors touch: what one tile drops near
    # a cut edge, its neighbour holds well inside. Without this the policy loses holes.
    places = yolo.origins(pistol.canon_size_px, tile, stride)

    assert places[0] == 0 and places[-1] == pistol.canon_size_px - tile, places
    assert all(b + margin <= a + tile - margin for a, b in zip(places, places[1:], strict=False)), (
        places
    )

    def finding(*points: tuple[float, float], confidence: float = 0.9):
        """A model that sees every one of ``points`` in whichever tile contains it."""

        def predict(crops, _settings):
            if not crops:  # the load probe check() makes; there is nothing here to load
                return []

            offsets = [(x, y) for x, y, _crop in yolo.tiles(target, pistol)]

            assert len(offsets) == len(crops), (len(offsets), len(crops))

            return [
                [
                    (px - ox - 6, py - oy - 6, px - ox + 6, py - oy + 6, confidence)
                    for px, py in points
                    if 0 <= px - ox < tile and 0 <= py - oy < tile
                ]
                for ox, oy in offsets
            ]

        return predict

    with tempfile.TemporaryDirectory() as tmp:
        weights = Path(tmp) / "yolov8n-target-v3.pt"
        weights.write_bytes(b"0")  # never opened: the predictor is injected
        run_output = weights.with_name("best.pt")
        run_output.write_bytes(b"0")
        trained = _settings(yolo_weights=weights)
        read = yolo.detect(target, pistol, settings=trained, predict=finding(*punched))

        # Every hole is cut into as many as four tiles, so this is the seam policy and the
        # overlap merge as much as it is the tile-to-canonical arithmetic.
        assert len(read) == len(punched), (len(read), len(punched))

        for px, py in punched:
            nearest = min(hypot(hit.x_canon - px, hit.y_canon - py) for hit in read)

            assert nearest < 0.5, (px, py, nearest)

        assert all(0 < hit.confidence <= 1 for hit in read), read
        assert [hit.confidence for hit in read] == sorted(
            (hit.confidence for hit in read), reverse=True
        )
        assert yolo.model_name(trained) == "yolov8n-target-v3", yolo.model_name(trained)
        assert yolo.detect(target, pistol, settings=trained, predict=finding()) == []

        try:
            yolo.detect(target, load_profile(PROFILE_JSON), settings=trained, predict=finding())
            raise AssertionError("an image that is not the canonical square must be refused")
        except ValueError as exc:
            assert not isinstance(exc, DetectorError), "a wrong frame is a bug, not a bad day"
            assert "canonical square" in str(exc), str(exc)

        for settings, hint in (
            (_settings(), "TA_YOLO_WEIGHTS"),
            # Blanked the way TA_SERVER_URL= is.
            (_settings(yolo_weights=""), "TA_YOLO_WEIGHTS"),
            (_settings(yolo_weights=Path(tmp) / "gone.pt"), "gone.pt"),
            (_settings(yolo_weights=run_output), "rename"),
        ):
            try:
                yolo.check(settings, pistol, predict=finding())
                raise AssertionError(f"{hint} must be refused")
            except DetectorError as exc:
                assert hint in str(exc), str(exc)

        try:
            scaleless = pistol.model_copy(update={"target_diam_mm": None})
            yolo.check(trained, scaleless, predict=finding())
            raise AssertionError("a profile with no scale must be refused")
        except DetectorError as exc:
            assert "target_diam_mm" in str(exc), str(exc)

    # --- the dataset a model is trained on, and the ways a split lies ---
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def marked(digest: str, image: np.ndarray, epoch: int = 1790000000) -> Path:
            """A hand-marked session folder for ``digest``, holes where _clean_target put them."""
            folder = root / f"{epoch}-{digest[:12]}-manual"
            folder.mkdir()
            cv2.imwrite(str(folder / "normalized.png"), image)
            (folder / "payload.json").write_text(
                build_payload(
                    image_sha256=digest,
                    profile=pistol,
                    registration=Registration(np.eye(3), target, None, 4, 4, "manual"),
                    hits=[Hit(x_canon=x, y_canon=y) for x, y in punched],
                    session=SessionMeta(
                        gun="G", distance_m=15, target_profile=pistol.name, target_profile_version=1
                    ),
                ).model_dump_json(),
                encoding="utf-8",
            )

            return folder

        held, kept = marked("a" * 64, target), marked("e" * 64, target)
        sessions = [held, kept]
        dataset = root / "ds"
        yolo_data.build(sessions, dataset, pistol, set())
        yolo_data.build(sessions, dataset, pistol, {"a" * 12})
        trained_on = {p.name.split("_")[0] for p in (dataset / "images" / "train").iterdir()}
        validated_on = {p.name.split("_")[0] for p in (dataset / "images" / "val").iterdir()}

        # The tiles of run one must be gone, or the photo held out of training is still in
        # it and every "measured on unseen paper" number after that is memorisation.
        assert trained_on == {"e" * 12}, f"a rebuilt split leaks: {trained_on}"
        assert validated_on == {"a" * 12}, validated_on

        # The client extra has no YAML parser, so the quoting is proven through json.
        odd = root / "runs: v2 #best"
        yolo_data.build(sessions, odd, pistol, set())
        line = (odd / "data.yaml").read_text(encoding="utf-8").splitlines()[0]

        assert json.loads(line.removeprefix("path: ")) == str(odd.resolve()), line

        for val, hint in (({"b" * 12}, "b" * 12), ({"a" * 12, "e" * 12}, "nothing to train")):
            try:
                yolo_data.build(sessions, dataset, pistol, val)
                raise AssertionError(f"{val} must be refused")
            except SystemExit as exc:
                assert hint in str(exc), str(exc)

        # Two markings of one photo can disagree, so the later one trains — and says so.
        later = marked("a" * 64, target, epoch=1790000009)
        heard = io.StringIO()

        with contextlib.redirect_stderr(heard):
            chosen = yolo_data._readings([held, later], pistol)["a" * 12][0]

        assert chosen == later, chosen
        assert f"{held.name}: superseded" in heard.getvalue(), heard.getvalue()

        # Clearing is how the split stays honest, so it must reach nothing it did not write.
        theirs = root / "theirs"
        (theirs / "images").mkdir(parents=True)
        (theirs / "images" / "keep.png").write_bytes(b"theirs")

        try:
            yolo_data.build(sessions, theirs, pistol, set())
            raise AssertionError("a directory this tool did not write must not be cleared")
        except SystemExit as exc:
            assert "refusing to clear" in str(exc), str(exc)

        assert (theirs / "images" / "keep.png").is_file(), "the clear reached somebody's files"

        # A rebuild that cannot happen must leave the dataset already there.
        validation = dataset / "images" / "val"
        before = sorted(p.name for p in validation.iterdir())
        crafted = marked("f" * 64, target)
        text = (crafted / "payload.json").read_text(encoding="utf-8")
        (crafted / "payload.json").write_text(
            text.replace("f" * 64, "../../../../etc/x"), encoding="utf-8"
        )

        for unusable in (marked("c" * 64, target[:100, :100]), crafted):
            try:
                yolo_data.build([unusable], dataset, pistol, set())
                raise AssertionError(f"{unusable.name} must be refused")
            except SystemExit as exc:
                assert "no usable" in str(exc), str(exc)

            after = sorted(p.name for p in validation.iterdir()) if validation.is_dir() else []

            assert after == before, f"{unusable.name}: a refused rebuild emptied the dataset"

        assert not list(root.glob("*_r0c0.png")), "a crafted digest wrote outside the dataset"

    assert sorted(
        [Path("10-aaaaaaaaaaaa-manual"), Path("2-aaaaaaaaaaaa-manual")], key=yolo_data._epoch
    ) == [Path("2-aaaaaaaaaaaa-manual"), Path("10-aaaaaaaaaaaa-manual")]

    # The lazy import is the contract: this module must stay usable, and this self-check
    # must stay runnable, on a machine that has only the client extra installed.
    assert "torch" not in sys.modules, "yolo imported torch without being asked for a model"

    # --- the printable sheet is to scale ---
    svg = render_svg(profile)
    assert 'width="175.500mm"' in svg  # 155.5 mm of target plus the two 10 mm margins
    assert svg.count("<circle") == profile.n_rings
    assert svg.count("<g transform") == len(profile.board.markers) + 1  # +1 for the margin shift

    # --- the outbox drains in order, exactly once, and survives the server going away ---
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # A first run: the outbox is created by the first enqueue, not before it.
        assert ship.pending(root / "never-created") == []

        order = root / "order"
        # Hand-named rather than enqueued, which would stamp three near-identical epochs.
        _queue(order, "2-aaaaaaaaaaaa-manual", "10-bbbbbbbbbbbb-manual", "1-cccccccccccc-manual")
        # The two things that share the directory without being sessions.
        (order / ".10-bbbbbbbbbbbb-manual-x9k2").mkdir()
        (order / "failed").mkdir()
        oldest_first = [
            "1-cccccccccccc-manual",
            "2-aaaaaaaaaaaa-manual",
            "10-bbbbbbbbbbbb-manual",
        ]
        assert [p.name for p in ship.pending(order)] == oldest_first, ship.pending(order)

        settings = _settings(server_url="https://example.invalid/", ingest_token=FAKE_TOKEN)
        sent: list[str] = []
        replies = iter((201, 201, 409))

        def accepting(_client, folder: Path, _settings) -> SimpleNamespace:
            sent.append(folder.name)

            return _reply(next(replies))

        assert ship.flush(order, settings=settings, send=accepting) == 0
        assert sent == oldest_first, sent
        # The third reply was a 409.
        assert ship.pending(order) == [], ship.pending(order)

        offline = root / "offline"
        _queue(
            offline, "20-dddddddddddd-manual", "21-eeeeeeeeeeee-manual", "22-ffffffffffff-manual"
        )
        tried: list[str] = []

        def drops(_client, folder: Path, _settings) -> SimpleNamespace:
            tried.append(folder.name)

            if len(tried) == 2:
                raise httpx2.ConnectError("server went away")

            return _reply(201)

        assert ship.flush(offline, settings=settings, send=drops) == 2
        assert tried == ["20-dddddddddddd-manual", "21-eeeeeeeeeeee-manual"], tried
        # The one that failed is kept and the one behind it was never attempted, so the
        # next run resends in the same order rather than skipping a session.
        assert [p.name for p in ship.pending(offline)] == [
            "21-eeeeeeeeeeee-manual",
            "22-ffffffffffff-manual",
        ]

        # A 500 says the server is broken, not this folder — the same answer as a dropped
        # connection, and the one an unhandled exception in ingest actually produces.
        assert ship.flush(offline, settings=settings, send=lambda *_: _reply(500)) == 2
        assert [p.name for p in ship.pending(offline)] == [
            "21-eeeeeeeeeeee-manual",
            "22-ffffffffffff-manual",
        ]

        refused = root / "refused"
        _queue(refused, "40-111111111111-manual")

        # Setting one of these aside would empty the whole outbox into failed/, one refusal
        # at a time.
        stops = ((401, "not this folder"), (404, "not this folder"), (301, "TA_SERVER_URL"))

        for code, hint in stops:
            try:
                ship.flush(refused, settings=settings, send=lambda *_, c=code: _reply(c))
                raise AssertionError(f"{code} must stop shipping, not clear the folder")
            except ship.ShipError as exc:
                assert hint in str(exc), (code, str(exc))

            assert [p.name for p in ship.pending(refused)] == ["40-111111111111-manual"], code

        # A server behind the client's contract refuses detector readings; the hand-marked
        # ones behind them must still ship.
        aside = root / "aside"
        _queue(aside, "50-222222222222-yolo", "51-333333333333-manual")
        answers = iter((422, 201))

        assert ship.flush(aside, settings=settings, send=lambda *_: _reply(next(answers))) == 0
        assert (aside / "failed" / "50-222222222222-yolo").is_dir(), "a refusal was lost"
        assert ship.pending(aside) == [], "a refusal held back the reading behind it"

        # The ground truth refused — a size cap hits exactly the folder with the original.
        pair = root / "pair"
        _queue(pair, "60-444444444444-manual", "60-444444444444-yolo")
        reached: list[str] = []

        def capped(_client, folder: Path, _settings) -> SimpleNamespace:
            reached.append(folder.name)

            return _reply(413)

        assert ship.flush(pair, settings=settings, send=capped) == 0
        assert reached == ["60-444444444444-manual"], f"a detector overtook its truth: {reached}"
        assert [p.name for p in ship.failed(pair)] == [
            "60-444444444444-manual",
            "60-444444444444-yolo",
        ], "the pair must go aside together, so they come back together"

        # Re-marked before it ever shipped.
        remarked = root / "remarked"
        _queue(
            remarked, "100-555555555555-manual", "100-555555555555-yolo", "200-555555555555-manual"
        )

        assert [p.name for p in ship.pending(remarked)] == [
            "200-555555555555-manual",
            "100-555555555555-yolo",
        ], ship.pending(remarked)

        # Refused under a name failed/ already holds: neither copy is overwritten.
        again = root / "again"
        _queue(again, "70-666666666666-yolo")
        earlier = again / "failed" / "70-666666666666-yolo"
        earlier.mkdir(parents=True)
        (earlier / "note").write_text("the earlier refusal", encoding="utf-8")
        ship.flush(again, settings=settings, send=lambda *_: _reply(422))

        assert (earlier / "note").is_file(), "an earlier refusal was overwritten"
        assert [p.name for p in ship.pending(again)] == ["70-666666666666-yolo"], "one was lost"

        # Two readings of one photo by one method, and an unrelated third.
        deduped = root / "deduped"
        _queue(
            deduped, "30-abcdefabcdef-manual", "31-abcdefabcdef-manual", "32-0123456789ab-manual"
        )
        assert [p.name for p in ship.pending(deduped)] == [
            "31-abcdefabcdef-manual",
            "32-0123456789ab-manual",
        ]
        assert not (deduped / "30-abcdefabcdef-manual").exists()

        # Two *methods* of one photo are two readings to compare, so neither supersedes the
        # other — the case a key of photo alone would silently delete half of.
        both = root / "both"
        _queue(both, "33-abcdefabcdef-manual", "34-abcdefabcdef-cv_blob")
        assert len(ship.pending(both)) == 2, ship.pending(both)

        # A tie on the epoch, which is what one run of the client produces.
        tied = root / "tied"
        _queue(tied, "35-abcdefabcdef-cv_blob", "35-abcdefabcdef-manual")
        assert [_parts(p).method for p in ship.pending(tied)] == ["manual", "cv_blob"]

        # A folder with no method in its name.
        legacy = root / "legacy"
        _queue(legacy, "36-abcdefabcdef")
        assert [p.name for p in ship.pending(legacy)] == ["36-abcdefabcdef"]

        without = ship.enqueue(root / "without", payload, b"png", None)
        assert not (without / "original.jpg").exists()
        assert ship.pending(root / "without") == [without]
        assert without.name.endswith(f"-{payload.image_sha256[:12]}-{payload.method}"), without.name
        complete = ship.enqueue(root / "with", payload, b"png", b"jpg")
        assert (complete / "original.jpg").is_file()

        # --- the request the sender builds, without a server to send it to ---
        request = ship.build_request(complete, settings)
        # A multipart request streams, so the body only exists once it is read.
        body = request.read()
        assert request.url.path == "/api/ingest"
        assert request.headers["authorization"] == f"Bearer {FAKE_TOKEN}"
        assert b'name="payload"' in body
        assert b'name="payload"; filename=' not in body, "payload must not be a file part"
        assert b'name="normalized"; filename="normalized.png"' in body
        assert b'name="original"; filename="original.jpg"' in body
        assert b'name="original"' not in ship.build_request(without, settings).read()

        # None of these can ship, and without this guard each would only say so after a
        # whole session has been clicked.
        for url, token in (
            ("localhost:8000", FAKE_TOKEN),
            ("http://", FAKE_TOKEN),
            ("https://ok.invalid", ""),
        ):
            try:
                ship.check_config(_settings(server_url=url, ingest_token=token))
                raise AssertionError(f"{url!r}/{token!r} must be refused up front")
            except ship.ShipError:
                pass

    print("client self-check OK")


if __name__ == "__main__":
    # Stripped here rather than inside selfcheck(), which is importable: a caller may be
    # relying on the TA_* vars this would otherwise delete out from under it.
    for key in [k for k in os.environ if k.startswith("TA_")]:
        del os.environ[key]

    selfcheck()
