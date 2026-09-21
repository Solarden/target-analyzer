"""Hole detection by a vision model. See implementation.md §9 and §13.

One OpenAI-compatible `/chat/completions` call with the canonical render attached, asking
for hole coordinates as JSON. The model is reached over plain HTTP, so this adds no
dependency — `httpx2` already ships the sessions.

Where this is expected to be weak, and why it is still worth measuring: a vision encoder
resamples the image to its own resolution, and a hole is about 17 px across in a 1500 px
frame. At a typical 896 px encoder that is ten pixels, and at 448 it is five. So precise
coordinates are the thing to doubt, not the thing to rely on — the brief frames this as a
comparison against the hand-marked reading rather than a replacement for it.

Where it should beat a blob filter is the question a blob filter cannot ask: *what is
this?* Printed digits, ring lines and patch tape all answer a size filter the same way a
hole does, and they are named in the prompt for exactly that reason.

No confidence is requested. A model will happily emit one and it would not be calibrated
against anything, and an invented number is worse in a comparison than an absent one.
"""

import argparse
import base64
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import cv2
import httpx2
import numpy as np

from ta_client.config import LOCAL, Settings, get_settings
from ta_client.detector import DetectorError
from ta_shared.agreement import MATCH_TOL_MM, agreement
from ta_shared.payload import MAX_HITS, Hit
from ta_shared.profile import TargetProfile, load_profile, mm_per_px

# Deterministic, because a detector that answers differently on the same photo cannot be
# compared against anything — including its own previous run.
TEMPERATURE = 0.0
# Separate from the read timeout: an unreachable host should cost seconds, not the whole
# budget a cold model legitimately needs.
CONNECT_TIMEOUT = 5.0

PROMPT = """This image is a {canon}x{canon} pixel photograph of a paper shooting target,
already squared up so the target fills the frame.

Find every bullet hole. A bullet hole is a torn puncture through the paper. Do NOT report
the printed scoring rings, the printed digits, or the white patch tape that covers holes
from earlier strings — those are markings, not holes.

Answer with JSON and nothing else, in this exact shape:
{{"holes": [{{"x": <pixels from the left edge>, "y": <pixels from the top edge>}}]}}"""


class VlmError(DetectorError):
    """The model could not be asked, or answered something that is not a reading."""


def model_name(settings: Settings | None = None) -> str | None:
    """Which model produced a reading, for the `model` column on the interpretation.

    Takes the same settings ``detect`` was given, so the name recorded beside a reading
    is the one that answered rather than whatever the environment says now.
    """
    return (settings or get_settings()).vlm_model or None


def check(settings: Settings | None, _profile: TargetProfile) -> None:
    """Refuse a run that cannot reach a model, before any clicking is done."""
    settings = settings or get_settings()

    if not settings.vlm_base_url:
        raise VlmError("TA_VLM_BASE_URL is not set — point it at an OpenAI-compatible endpoint")

    if not settings.vlm_model:
        raise VlmError("TA_VLM_MODEL is not set — name the model to ask")

    url = urlparse(settings.vlm_base_url)

    # The two guards ship.check_config makes for the ingest token, and for its reasons.
    if url.scheme not in ("http", "https") or not url.hostname:
        raise VlmError(
            f"TA_VLM_BASE_URL must be a full http:// or https:// URL — "
            f"got {settings.vlm_base_url!r}"
        )

    # Refused rather than warned about, unlike the ingest token: the endpoint this is
    # built for is TLS, so plain http with a key set is a mistake rather than a choice.
    if (
        url.scheme == "http"
        and url.hostname not in LOCAL
        and settings.vlm_api_key.get_secret_value()
    ):
        raise VlmError(
            f"TA_VLM_API_KEY is set but {url.hostname} is plain http, so the key would "
            "cross the network in the clear — use https, or drop the key"
        )


def build_request(image_png: bytes, canon: int, settings: Settings) -> httpx2.Request:
    """The one call, built separately so it can be read without a model to send it to."""
    encoded = base64.b64encode(image_png).decode("ascii")
    headers = {"Content-Type": "application/json"}

    # Sent only when set: a local endpoint usually has no auth, and an empty Bearer reads
    # as a malformed credential rather than as no credential.
    if settings.vlm_api_key.get_secret_value():
        headers["Authorization"] = f"Bearer {settings.vlm_api_key.get_secret_value()}"

    return httpx2.Request(
        "POST",
        f"{settings.vlm_base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json={
            "model": settings.vlm_model,
            "temperature": TEMPERATURE,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT.format(canon=canon)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
        },
    )


Sender = Callable[[httpx2.Client, httpx2.Request], httpx2.Response]


def post(client: httpx2.Client, request: httpx2.Request) -> httpx2.Response:
    return client.send(request)


def parse_hits(answer: str, canon: int) -> tuple[list[Hit], int]:
    """The hole coordinates the model gave, and how many of them landed off the frame."""
    # Fenced first. The brace span below runs from the first "{" to the last, which is
    # right for nested objects and wrong when the prose ahead of it carries a brace.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", answer, re.DOTALL)
    match = fenced.group(1) if fenced else None

    if match is None:
        span = re.search(r"\{.*\}", answer, re.DOTALL)
        match = span.group() if span else None

    if match is None:
        raise VlmError(f"no JSON object in the model's answer: {answer[:200]!r}")

    try:
        holes = json.loads(match)["holes"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise VlmError(f"answer is not an object with a 'holes' key: {exc}") from exc

    # An empty list is a real answer — the model saw no holes. A shape we cannot read is
    # not, and skipping it would report the two identically.
    if not isinstance(holes, list):
        raise VlmError(f"'holes' is {type(holes).__name__}, not a list")

    hits, off_frame = [], 0

    for hole in holes:
        try:
            x, y = float(hole["x"]), float(hole["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VlmError(f"{hole!r} is not a hole: {exc}") from exc

        # Counted rather than refused: a model guessing at the scale it was handed is a
        # behaviour worth measuring, and a guess off the paper is not a hole regardless.
        if 0 <= x <= canon and 0 <= y <= canon:
            hits.append(Hit(x_canon=x, y_canon=y))
        else:
            off_frame += 1

    return hits[:MAX_HITS], off_frame


def detect(
    normalized: np.ndarray,
    profile: TargetProfile,
    *,
    settings: Settings | None = None,
    send: Sender = post,
) -> list[Hit]:
    """Ask a vision model for the holes in a canonical-frame image.

    Raises ``ValueError`` when the image is not the profile's canonical square, and
    ``VlmError`` when the model cannot be reached or does not answer with a reading.
    """
    canon = profile.canon_size_px

    # The model is told the frame size in the prompt, so an image that is not that size
    # would have it scaling its answer against a square that is somewhere else.
    if normalized.ndim != 3 or normalized.shape[:2] != (canon, canon):
        shape = "x".join(str(n) for n in normalized.shape)

        raise ValueError(
            f"image is {shape}, not {profile.name} v{profile.version}'s "
            f"{canon}x{canon}x3 canonical square"
        )

    settings = settings or get_settings()
    check(settings, profile)
    ok, buffer = cv2.imencode(".png", normalized)

    if not ok:
        raise VlmError("could not encode the normalized image")

    request = build_request(buffer.tobytes(), canon, settings)
    timeout = httpx2.Timeout(settings.vlm_timeout, connect=CONNECT_TIMEOUT)

    try:
        with httpx2.Client(timeout=timeout) as client:
            response = send(client, request)
    except httpx2.RequestError as exc:
        raise VlmError(f"{type(exc).__name__}: {exc}") from exc

    if response.status_code != 200:
        raise VlmError(f"the model answered {response.status_code}: {response.text[:200]}")

    try:
        answer = response.json()["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise VlmError(f"not an OpenAI-compatible reply: {exc}") from exc

    # Some servers answer with a list of content parts rather than a string. Refused with
    # the type named, so the message says what to add if this is ever the real shape.
    if not isinstance(answer, str):
        raise VlmError(f"the model's content is {type(answer).__name__}, not text")

    hits, off_frame = parse_hits(answer, canon)

    if off_frame:
        # The failure this method is most expected to have: a vision encoder resamples the
        # frame, so the model can answer in a square that is not the one it was told about.
        print(f"vlm: {off_frame} coordinates outside the frame, dropped", file=sys.stderr)

    return hits


def _report(folder: Path, payload: dict, profile: TargetProfile) -> None:
    """Measure one stored session: what the model finds against what a person marked."""
    normalized = cv2.imread(str(folder / "normalized.png"))

    if normalized is None:
        raise SystemExit(f"no readable normalized.png in {folder}")

    truth = [(hit["x_canon"], hit["y_canon"]) for hit in payload["hits"]]

    try:
        proposal = detect(normalized, profile)
    except (DetectorError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    result = agreement(
        truth,
        [(hit.x_canon, hit.y_canon) for hit in proposal],
        MATCH_TOL_MM / mm_per_px(profile),
    )
    offset = "—" if result.mean_offset_px is None else f"{result.mean_offset_px:.1f} px"

    print(f"{folder.name}: {len(truth)} marked by hand, {len(proposal)} proposed")
    print(f"  matched {result.matched} · missed {result.missed} · spurious {result.spurious}")
    print(f"  mean offset over the matched {offset}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="ta_client.vlm", description="Score the vision model against a marked session."
    )
    parser.add_argument(
        "session", type=Path, help="a session folder: payload.json + normalized.png"
    )
    parser.add_argument(
        "--profile", type=Path, default=None, help="path to the target profile JSON"
    )
    args = parser.parse_args(argv)
    marked = args.session / "payload.json"

    if not marked.is_file():
        raise SystemExit(f"no payload.json in {args.session}")

    payload = json.loads(marked.read_text(encoding="utf-8"))
    profile_path = args.profile or (
        Path(__file__).resolve().parents[2]
        / "profiles"
        / f"{payload['session']['target_profile']}.json"
    )

    if not profile_path.is_file():
        raise SystemExit(f"no profile at {profile_path} — pass --profile")

    _report(args.session, payload, load_profile(profile_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
