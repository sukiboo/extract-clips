import os
import re
import subprocess
import sys
from datetime import datetime

import static_ffmpeg

from src.constants import INPUT_DIR, OUTPUT_DIR, VIDEO_EXTENSIONS

# Download ffmpeg binary if needed (runs once at module import)
static_ffmpeg.add_paths()


def list_video_files() -> list[str]:
    """Find all video files in the input directory."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if os.path.isfile(os.path.join(INPUT_DIR, f))
        and os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        print(f"No video files found in {INPUT_DIR}/")
        print("Supported formats: " + ", ".join(VIDEO_EXTENSIONS))
    else:
        print(f"Found {len(video_files)} videos to process in `{INPUT_DIR}`...\n")

    return video_files


def get_video_start_time(video_path: str) -> datetime:
    """Get the video start time from filename or file modification time.

    Tries to parse timestamp from filename patterns like:
        Ring_20250919_1734_...  -> 2025-09-19 17:34:00

    Falls back to file modification time if parsing fails.

    Args:
        video_path: Path to the video file.

    Returns:
        datetime of when the video started.
    """
    filename = os.path.basename(video_path)

    # Try to match Ring camera format: Ring_YYYYMMDD_HHMM_...
    match = re.search(r"(\d{8})_(\d{4})", filename)
    if match:
        date_str, time_str = match.groups()
        try:
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
        except ValueError:
            pass

    # Fall back to file modification time
    return datetime.fromtimestamp(os.path.getmtime(video_path))


def extract_clip(input_path: str, output_path: str, start: float, end: float) -> bool:
    """Extract a clip using ffmpeg without re-encoding (fast).

    Args:
        input_path: Path to the input video file.
        output_path: Path to the output video file.
        start: Start time in seconds.
        end: End time in seconds.

    Returns:
        True on success, False on failure.
    """
    duration = end - start

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-ss",
        f"{start:.3f}",  # Start time
        "-i",
        input_path,  # Input file
        "-t",
        f"{duration:.3f}",  # Duration
        "-c",
        "copy",  # Copy streams (no re-encode)
        "-avoid_negative_ts",
        "make_zero",
        output_path,
    ]

    try:
        _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("  Error: ffmpeg not found. Please install ffmpeg.")
        sys.exit(1)
