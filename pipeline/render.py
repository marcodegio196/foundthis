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


VERTICAL_ASPECT = 9 / 16


def _even(value: float) -> int:
    """Nearest even integer — odd dimensions break yuv420p encoding."""
    rounded = int(round(value))
    return max(2, rounded - (rounded % 2))


def vertical_output_size(output_height: int) -> tuple[int, int]:
    """Exact 9:16 output dimensions, e.g. 1920 -> (1080, 1920)."""
    return _even(output_height * VERTICAL_ASPECT), _even(output_height)


def crop_to_vertical(width: int, height: int) -> str | None:
    """Centre-crop filter bringing a source to 9:16, or None if it already is.

    Cropping 16:9 to 9:16 keeps only the middle ~32% of the frame, so a shot
    whose subject sits off-centre is better served by a native 9:16 master —
    which is why the selection query prefers those.

    The crop cannot always land on exactly 9:16: a 720-tall source wants a
    405-wide crop, and odd widths break yuv420p. The residual fraction of a
    percent is taken out by the scale filter, which pins the final size.
    """
    if width <= 0 or height <= 0:
        return None
    ratio = width / height
    if abs(ratio - VERTICAL_ASPECT) < 0.001:
        return None
    if ratio > VERTICAL_ASPECT:  # landscape or square: trim the sides
        return f"crop={min(_even(height * VERTICAL_ASPECT), _even(width))}:{_even(height)}"
    # Taller than 9:16 (rare): trim top and bottom instead of overcropping.
    return f"crop={_even(width)}:{min(_even(width / VERTICAL_ASPECT), _even(height))}"


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
    out_width, out_height = vertical_output_size(output_height)
    filters = [f for f in (crop_to_vertical(width, height),) if f]
    # Pinned to exact dimensions rather than derived with -2: a source whose
    # crop can't land on exactly 9:16 would otherwise produce something like
    # 1078x1920, which platforms re-encode. setsar=1 stops a non-square pixel
    # aspect surviving from the source and skewing playback.
    filters.append(f"scale={out_width}:{out_height},setsar=1")

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
