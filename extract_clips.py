#!/usr/bin/env python3
"""
Motion Clip Extractor

Detects motion in Ring camera footage and extracts clips containing movement.
Ideal for monitoring pets in mostly-static video.
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta

import cv2
import static_ffmpeg

# Download ffmpeg binary if needed
static_ffmpeg.add_paths()

# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = "./videos/inputs"
OUTPUT_DIR = "./videos/outputs"

MOTION_THRESHOLD = 1000  # Min contour area to count as motion
MIN_CLIP_DURATION = 1.0  # Seconds - ignore very brief motion
MERGE_GAP = 2.0  # Seconds - merge motion events closer than this
BUFFER_BEFORE = 1.0  # Seconds to include before motion starts
BUFFER_AFTER = 1.0  # Seconds to include after motion ends

FRAME_SKIP = 3  # Process every Nth frame for speed
DEBUG = True  # Print debug info about detected motion

# Background subtractor settings (MOG2 algorithm)
# Docs: https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
BG_HISTORY = 500  # Frames used to build background model
BG_VAR_THRESHOLD = 50  # Variance threshold for background/foreground
BG_DETECT_SHADOWS = False  # Whether to detect and mark shadows

FALLBACK_FPS = 30.0  # FPS to use if video metadata is missing
DEBUG_MIN_AREA = 500  # Min contour area to include in debug samples
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# =============================================================================
# Motion Detection
# =============================================================================


def detect_motion_timestamps(video_path: str) -> list[float]:
    """
    Detect timestamps (in seconds) where motion occurs in the video.
    Uses background subtraction optimized for static cameras.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Could not open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = FALLBACK_FPS

    # Background subtractor - good for static cameras
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESHOLD, detectShadows=BG_DETECT_SHADOWS
    )

    motion_timestamps = []
    frame_idx = 0
    max_area_seen = 0
    area_samples: list[tuple[float, float]] = []  # For debug output

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
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find largest contour area in this frame
        frame_max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            frame_max_area = max(frame_max_area, area)

        if frame_max_area > max_area_seen:
            max_area_seen = frame_max_area

        # Sample some areas for debug output
        if DEBUG and frame_max_area > DEBUG_MIN_AREA and len(area_samples) < 20:
            timestamp = frame_idx / fps
            area_samples.append((timestamp, frame_max_area))

        # Check if motion exceeds threshold
        if frame_max_area > MOTION_THRESHOLD:
            timestamp = frame_idx / fps
            motion_timestamps.append(timestamp)

        frame_idx += 1

    cap.release()

    if DEBUG:
        print(f"  Max contour area seen: {max_area_seen:.0f} (threshold: {MOTION_THRESHOLD})")
        if area_samples:
            print(f"  Sample detections (time, area):")
            for ts, area in area_samples[:10]:
                marker = " <-- above threshold" if area > MOTION_THRESHOLD else ""
                print(f"    {ts:.1f}s: area={area:.0f}{marker}")

    return motion_timestamps


# =============================================================================
# Timestamp Merging
# =============================================================================


def merge_timestamps_into_ranges(
    timestamps: list[float], video_duration: float
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


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using OpenCV."""
    cap = cv2.VideoCapture(video_path)
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


# =============================================================================
# Main Processing
# =============================================================================


def process_video(video_path: str) -> int:
    """
    Process a single video: detect motion, merge timestamps, extract clips.
    Returns the number of clips extracted.
    """
    video_name = os.path.basename(video_path)
    print(f"\nProcessing: {video_name}")

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
    video_start_time = datetime.fromtimestamp(os.path.getmtime(video_path))

    for i, (start, end) in enumerate(ranges, 1):
        # Calculate actual clock time when motion occurred
        motion_time = video_start_time + timedelta(seconds=start)
        time_str = motion_time.strftime("%Y-%m-%d_%H-%M-%S")
        output_name = f"{time_str}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        print(f"  Extracting clip {i}: {start:.1f}s - {end:.1f}s")

        if extract_clip(video_path, output_path, start, end):
            clips_extracted += 1
            print(f"    Saved: {output_name}")

    return clips_extracted


def main():
    """Main entry point."""
    # Ensure directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all video files
    video_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if os.path.isfile(os.path.join(INPUT_DIR, f))
        and os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        print(f"No video files found in {INPUT_DIR}/")
        print("Supported formats: " + ", ".join(VIDEO_EXTENSIONS))
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
