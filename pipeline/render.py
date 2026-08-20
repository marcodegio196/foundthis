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


# The overlay sits in the lower third. Text is drawn from `OVERLAY_BASE_Y`
# downward, so this is the band it actually covers.
OVERLAY_BASE_Y = 0.72
OVERLAY_BAND = (0.70, 0.85)

# The band is measured as a coarse grid rather than a single average, because
# the average is the wrong question. Text fails where it crosses a bright patch,
# not where the backdrop is bright on average, and a sunlit rock beside dark
# water averages out to something unremarkable. Measured on a clip whose
# headline was genuinely unreadable against white rock, against three that read
# fine:
#
#              mean    80th pct   brightest
#   unreadable  0.51      0.63       0.84
#   fine        0.36      0.43       0.57
#   fine        0.41      0.50       0.58
#   fine        0.31      0.44       0.62
#
# The mean puts the failure between two of the passes. The 80th percentile
# separates it cleanly, which is what a reader notices: a fifth of the strip
# being bright is enough to lose the glyphs crossing it.
OVERLAY_GRID = (8, 2)
BRIGHT_PERCENTILE = 0.80
LIGHT_BACKDROP = 0.60


def build_overlay_luma_command(
    source: str | Path,
    *,
    in_point: float,
    out_point: float,
    width: int,
    height: int,
    sample_fps: float = 2.0,
    band: tuple[float, float] = OVERLAY_BAND,
) -> list[str]:
    """Emit one grey byte per sampled frame: the mean brightness behind the text.

    The crop chain matches the social render exactly, so what is measured is
    what ends up under the overlay rather than the middle of the uncropped
    source. `scale=1:1` makes ffmpeg average the region down to a single pixel,
    which avoids decoding frames in Python for a number this small.

    Horizontally it samples the middle 70%: the text is centred, and the outer
    edges are backdrop the reader never sees behind a glyph.
    """
    top, bottom = band
    filters = [f for f in (crop_to_vertical(width, height),) if f]
    filters.append(
        f"crop=iw*0.7:ih*{bottom - top:.3f}:iw*0.15:ih*{top:.3f}"
    )
    columns, rows = OVERLAY_GRID
    filters += [f"fps={sample_fps}", f"scale={columns}:{rows}", "format=gray"]
    return [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-ss", f"{in_point:.3f}", "-to", f"{out_point:.3f}", "-i", str(source),
        "-vf", ",".join(filters),
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]


def backdrop_brightness(raw: bytes) -> float | None:
    """How bright the light part of the backdrop is, 0..1.

    Takes the 80th percentile across every grid cell of every sampled frame,
    so one sunlit patch persisting through the clip registers even when the
    strip as a whole is mid-toned. Returns None when nothing was decoded.
    """
    if not raw:
        return None
    values = sorted(raw)
    index = min(int(len(values) * BRIGHT_PERCENTILE), len(values) - 1)
    return values[index] / 255.0


def overlay_palette(luma: float | None) -> tuple[str, str]:
    """Text and shadow colours for a backdrop of this brightness.

    White-on-bright is the failure this exists for: over sunlit rock the
    headline all but vanishes, and a drop shadow does not save it because the
    shadow becomes the only thing carrying the shape. Flipping to black text
    with a light shadow keeps the same weight without putting a box behind it.

    An unmeasurable clip keeps white, which is right far more often than not
    for aerial footage — sky, water, and foliage all sit well below the bar.
    """
    if luma is not None and luma > LIGHT_BACKDROP:
        return "black", "white@0.55"
    return "white", "black@0.55"


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
    font_color: str = "white",
    shadow_color: str = "black@0.55",
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
    base_y = OVERLAY_BASE_Y
    for index, line in enumerate(lines):
        options = [
            f"text='{escape_drawtext(line)}'",
            f"fontcolor={font_color}",
            f"fontsize=h/{22 if index else 16}",
            "x=(w-text_w)/2",
            f"y=h*{base_y + index * 0.06:.3f}",
            "box=0",
            f"shadowcolor={shadow_color}",
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
