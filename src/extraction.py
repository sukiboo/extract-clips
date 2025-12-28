import subprocess
import sys

import static_ffmpeg

# Download ffmpeg binary if needed (runs once at module import)
static_ffmpeg.add_paths()


def extract_clip(input_path: str, output_path: str, start: float, end: float) -> bool:
    """
    Extract a clip using ffmpeg without re-encoding (fast).
    Returns True on success.
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
