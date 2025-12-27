#!/usr/bin/env python3
"""
Motion Clip Extractor

Detects motion in Ring camera footage and extracts clips containing movement.
Ideal for monitoring pets in mostly-static video.
"""

import subprocess
import sys
from pathlib import Path

import cv2

# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = Path("videos/inputs")
OUTPUT_DIR = Path("videos/outputs")

MOTION_THRESHOLD = 5000      # Min contour area to count as motion
MIN_CLIP_DURATION = 1.0      # Seconds - ignore very brief motion
MERGE_GAP = 2.0              # Seconds - merge motion events closer than this
BUFFER_BEFORE = 1.0          # Seconds to include before motion starts
BUFFER_AFTER = 1.0           # Seconds to include after motion ends

FRAME_SKIP = 3               # Process every Nth frame for speed

# =============================================================================
# Motion Detection
# =============================================================================

def detect_motion_timestamps(video_path: Path) -> list[float]:
    """
    Detect timestamps (in seconds) where motion occurs in the video.
    Uses background subtraction optimized for static cameras.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Could not open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Fallback

    # Background subtractor - good for static cameras
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False
    )

    motion_timestamps = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for speed
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)

        # Find contours in the foreground mask
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Check if any contour is large enough to be considered motion
        for contour in contours:
            if cv2.contourArea(contour) > MOTION_THRESHOLD:
                timestamp = frame_idx / fps
                motion_timestamps.append(timestamp)
                break  # One motion detection per frame is enough

        frame_idx += 1

    cap.release()
    return motion_timestamps


# =============================================================================
# Timestamp Merging
# =============================================================================

def merge_timestamps_into_ranges(
    timestamps: list[float],
    video_duration: float
) -> list[tuple[float, float]]:
    """
    Merge nearby motion timestamps into continuous time ranges.
    Adds buffer before/after and filters out very short clips.
    """
    if not timestamps:
        return []

    timestamps = sorted(timestamps)
    ranges = []
    range_start = timestamps[0]
    range_end = timestamps[0]

    for ts in timestamps[1:]:
        if ts - range_end <= MERGE_GAP:
            # Extend current range
            range_end = ts
        else:
            # Save current range and start new one
            ranges.append((range_start, range_end))
            range_start = ts
            range_end = ts

    # Don't forget the last range
    ranges.append((range_start, range_end))

    # Add buffers and filter short clips
    final_ranges = []
    for start, end in ranges:
        duration = end - start
        if duration < MIN_CLIP_DURATION:
            continue

        # Add buffers, clamped to video bounds
        buffered_start = max(0, start - BUFFER_BEFORE)
        buffered_end = min(video_duration, end + BUFFER_AFTER)
        final_ranges.append((buffered_start, buffered_end))

    return final_ranges


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0:
        return 0.0

    return frame_count / fps


# =============================================================================
# Clip Extraction
# =============================================================================

def extract_clip(
    input_path: Path,
    output_path: Path,
    start: float,
    end: float
) -> bool:
    """
    Extract a clip using ffmpeg without re-encoding (fast).
    Returns True on success.
    """
    duration = end - start

    cmd = [
        "ffmpeg",
        "-y",                    # Overwrite output
        "-ss", f"{start:.3f}",   # Start time
        "-i", str(input_path),   # Input file
        "-t", f"{duration:.3f}", # Duration
        "-c", "copy",            # Copy streams (no re-encode)
        "-avoid_negative_ts", "make_zero",
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("  Error: ffmpeg not found. Please install ffmpeg.")
        sys.exit(1)


# =============================================================================
# Main Processing
# =============================================================================

def process_video(video_path: Path) -> int:
    """
    Process a single video: detect motion, merge timestamps, extract clips.
    Returns the number of clips extracted.
    """
    print(f"\nProcessing: {video_path.name}")

    # Get video duration
    duration = get_video_duration(video_path)
    if duration <= 0:
        print("  Error: Could not determine video duration")
        return 0

    print(f"  Duration: {duration:.1f}s")

    # Detect motion
    print("  Detecting motion...")
    timestamps = detect_motion_timestamps(video_path)
    print(f"  Found {len(timestamps)} motion frames")

    if not timestamps:
        print("  No motion detected")
        return 0

    # Merge into ranges
    ranges = merge_timestamps_into_ranges(timestamps, duration)
    print(f"  Merged into {len(ranges)} clip(s)")

    if not ranges:
        print("  No clips long enough to extract")
        return 0

    # Extract clips
    clips_extracted = 0
    stem = video_path.stem

    for i, (start, end) in enumerate(ranges, 1):
        output_name = f"{stem}_clip{i}_{start:.1f}-{end:.1f}.mp4"
        output_path = OUTPUT_DIR / output_name

        print(f"  Extracting clip {i}: {start:.1f}s - {end:.1f}s")

        if extract_clip(video_path, output_path, start, end):
            clips_extracted += 1
            print(f"    Saved: {output_name}")

    return clips_extracted


def main():
    """Main entry point."""
    # Ensure directories exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    video_files = [
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"No video files found in {INPUT_DIR}/")
        print("Supported formats: " + ", ".join(video_extensions))
        return

    print(f"Found {len(video_files)} video(s) to process")

    # Process each video
    total_clips = 0
    for video_path in sorted(video_files):
        clips = process_video(video_path)
        total_clips += clips

    print(f"\n{'='*50}")
    print(f"Done! Extracted {total_clips} clip(s) to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
