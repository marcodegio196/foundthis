"""ffmpeg command construction for Stage 4.

Pure functions returning argument lists. The filter graphs are the part most
likely to be wrong and most annoying to debug through a 4K encode, so they are
built and asserted on without ever invoking ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

from .media import FFMPEG

# Both profiles cut from the same source with the same seek strategy: seek to
# the shot's in-point *before* -i for speed, then re-encode. Frame-accurate
# trimming rules out a stream copy, and a licensing master has to start exactly
# where the shot starts.
SOCIAL_CRF = 20
LICENSING_CRF = 12


def escape_drawtext(text: str) -> str:
    """Escape a string for ffmpeg's drawtext filter.

    drawtext parses `:` as an option separator and `'` as a quote, so a location
    like "Vlorë: old town" silently breaks the whole filter chain.
    """
    for char, replacement in (("\\", r"\\"), (":", r"\:"), ("'", r"\'"), ("%", r"\%")):
        text = text.replace(char, replacement)
    return text


def overlay_lines(
    year: int | str | None, country: str | None, site: str | None, headline: str
) -> list[str]:
    """The two overlay lines: the fixed headline, then 'year · location'."""
    location = " · ".join(part for part in (site, country) if part)
    subtitle = " · ".join(str(part) for part in (year, location) if part)
    return [headline, subtitle] if subtitle else [headline]


def crop_to_vertical(width: int, height: int) -> str | None:
    """Centre-crop filter taking a landscape source to 9:16, or None if it already is.

    Cropping 16:9 to 9:16 keeps only the middle ~32% of the frame, so a shot
    whose subject sits off-centre is better served by a native 9:16 master —
    which is why the selection query prefers those.
    """
    if width <= 0 or height <= 0:
        return None
    target = 9 / 16
    if abs(width / height - target) < 0.01:
        return None
    crop_width = int(height * target) // 2 * 2
    return f"crop={crop_width}:{height}"


def build_social_command(
    source: str | Path,
    output: str | Path,
    *,
    in_point: float,
    out_point: float,
    width: int,
    height: int,
    lines: list[str],
    font_file: str | Path | None = None,
    output_height: int = 1920,
) -> list[str]:
    """9:16, overlay burned in, compressed for platform delivery."""
    filters = [f for f in (crop_to_vertical(width, height),) if f]
    filters.append(f"scale=-2:{output_height}")

    # Lines are drawn from the lower third upward, the safe area on every
    # platform once captions and UI chrome are accounted for.
    base_y = 0.72
    for index, line in enumerate(lines):
        options = [
            f"text='{escape_drawtext(line)}'",
            "fontcolor=white",
            f"fontsize=h/{22 if index else 16}",
            "x=(w-text_w)/2",
            f"y=h*{base_y + index * 0.06:.3f}",
            "box=0",
            "shadowcolor=black@0.55",
            "shadowx=2",
            "shadowy=2",
        ]
        if font_file:
            options.insert(1, f"fontfile='{escape_drawtext(str(font_file))}'")
        filters.append("drawtext=" + ":".join(options))

    return [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{in_point:.3f}",
        "-to", f"{out_point:.3f}",
        "-i", str(source),
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", "slow", "-crf", str(SOCIAL_CRF),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",  # platform audio is added at post time, not burned in here
        str(output),
    ]


def build_licensing_command(
    source: str | Path,
    output: str | Path,
    *,
    in_point: float,
    out_point: float,
    codec: str = "libx264",
) -> list[str]:
    """Clean master: native resolution, no overlay, no watermark, audio kept."""
    command = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{in_point:.3f}",
        "-to", f"{out_point:.3f}",
        "-i", str(source),
        "-c:v", codec,
    ]
    if codec == "libx264":
        command += ["-preset", "slow", "-crf", str(LICENSING_CRF), "-pix_fmt", "yuv420p"]
    elif codec == "prores_ks":
        command += ["-profile:v", "3"]
    command += ["-c:a", "copy", str(output)]
    return command
