"""ffprobe/ffmpeg wrappers.

The only module that shells out to the ffmpeg toolchain. Parsing is kept in
pure functions so it can be tested against captured ffprobe output without the
binaries present.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

FFPROBE = "ffprobe"
FFMPEG = "ffmpeg"


class MediaError(RuntimeError):
    """ffprobe/ffmpeg failed or is missing."""


def toolchain_available() -> bool:
    return bool(shutil.which(FFPROBE) and shutil.which(FFMPEG))


def run(cmd: list[str], *, capture: bool = True) -> str:
    try:
        result = subprocess.run(
            cmd, capture_output=capture, text=True, check=True
        )
    except FileNotFoundError as exc:
        raise MediaError(f"{cmd[0]} not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "").strip().splitlines()[-3:]
        raise MediaError(f"{cmd[0]} failed: {' / '.join(tail)}") from exc
    return result.stdout if capture else ""


def run_bytes(cmd: list[str]) -> bytes:
    """Run a command and return stdout as bytes.

    `run` decodes as text, which corrupts anything binary. Used for filters
    that emit raw pixels rather than a report.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise MediaError(f"{cmd[0]} not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
        raise MediaError(f"{cmd[0]} failed: {' / '.join(tail)}") from exc
    return result.stdout


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def ffprobe_raw(path: str | Path) -> dict[str, Any]:
    out = run(
        [
            FFPROBE, "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ]
    )
    return json.loads(out)


def parse_fps(rate: str | None) -> float | None:
    """'30000/1001' -> 29.97. ffprobe also emits '0/0' for streams with no rate."""
    if not rate:
        return None
    try:
        value = float(Fraction(rate))
    except (ZeroDivisionError, ValueError):
        return None
    return round(value, 3) if value else None


def parse_rotation(stream: dict[str, Any]) -> int:
    """Rotation in degrees, from either the tag or the display-matrix side data.

    Phone and gimbal footage is routinely stored landscape with a rotation flag,
    so ignoring this mislabels 9:16 masters as 16:9.
    """
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return int(side_data["rotation"]) % 360
    tag = (stream.get("tags") or {}).get("rotate")
    return int(tag) % 360 if tag else 0


def aspect_label(width: int | None, height: int | None) -> str | None:
    """'16:9' / '9:16' when close enough, else the reduced ratio."""
    if not width or not height:
        return None
    ratio = width / height
    for label, target in (("16:9", 16 / 9), ("9:16", 9 / 16), ("4:3", 4 / 3),
                          ("1:1", 1.0), ("21:9", 21 / 9)):
        if math.isclose(ratio, target, rel_tol=0.02):
            return label
    fraction = Fraction(width, height).limit_denominator(50)
    return f"{fraction.numerator}:{fraction.denominator}"


_ISO6709 = re.compile(r"([+-]\d+\.?\d*)([+-]\d+\.?\d*)")


def parse_iso6709(value: str | None) -> tuple[float | None, float | None]:
    """'+41.3275+019.8187+123.456/' -> (41.3275, 19.8187)."""
    if not value:
        return None, None
    match = _ISO6709.match(value.strip())
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def parse_probe(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten ffprobe JSON into the columns `sources` stores."""
    fmt = raw.get("format") or {}
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    video = next(
        (s for s in raw.get("streams") or [] if s.get("codec_type") == "video"), {}
    )
    stream_tags = {k.lower(): v for k, v in (video.get("tags") or {}).items()}

    width, height = video.get("width"), video.get("height")
    if parse_rotation(video) in (90, 270):
        width, height = height, width

    lat, lon = parse_iso6709(
        tags.get("com.apple.quicktime.location.iso6709") or tags.get("location")
    )

    duration = fmt.get("duration") or video.get("duration")
    bitrate = fmt.get("bit_rate")
    make = tags.get("com.apple.quicktime.make") or tags.get("make")
    model = tags.get("com.apple.quicktime.model") or tags.get("model")

    return {
        "duration": float(duration) if duration else None,
        "width": width,
        "height": height,
        "fps": parse_fps(video.get("r_frame_rate")),
        "codec": video.get("codec_name"),
        "bitrate": int(bitrate) if bitrate else None,
        "aspect": aspect_label(width, height),
        "captured_at": tags.get("creation_time") or stream_tags.get("creation_time"),
        "gps_lat": lat,
        "gps_lon": lon,
        "drone_model": " ".join(p for p in (make, model) if p) or None,
    }


def probe(path: str | Path) -> dict[str, Any]:
    return parse_probe(ffprobe_raw(path))


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------


def sample_timestamps(
    in_point: float, out_point: float, fps: float, *, max_frames: int = 24
) -> list[float]:
    """Evenly spaced sample times inside a shot, endpoints trimmed.

    The first and last frames of a shot are the least representative (they sit
    on the cut), so sampling starts half an interval in.
    """
    duration = max(out_point - in_point, 0.0)
    if duration <= 0:
        return []
    count = max(1, min(int(duration * fps), max_frames))
    step = duration / count
    return [round(in_point + step * (i + 0.5), 3) for i in range(count)]


def extract_frames(
    path: str | Path,
    timestamps: list[float],
    out_dir: str | Path,
    *,
    width: int = 512,
) -> list[Path]:
    """Write one downscaled JPEG per timestamp. Returns the paths written.

    Seeks per frame rather than decoding the whole shot — for 4K sources that is
    the difference between the scoring pass taking hours and taking days.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, timestamp in enumerate(timestamps):
        target = out_dir / f"{index:04d}.jpg"
        run(
            [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{timestamp:.3f}", "-i", str(path),
                "-frames:v", "1",
                "-vf", f"scale={width}:-2",
                "-q:v", "3",
                str(target),
            ]
        )
        if target.exists():
            written.append(target)
    return written


def iter_gray_frames(
    path: str | Path, timestamps: list[float], *, width: int = 256
) -> Iterator[Any]:
    """Yield sampled frames as grayscale numpy arrays (needs opencv + numpy)."""
    import tempfile

    import cv2  # imported lazily so the DB layer stays dependency-free

    with tempfile.TemporaryDirectory() as tmp:
        for frame_path in extract_frames(path, timestamps, tmp, width=width):
            image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                yield image
