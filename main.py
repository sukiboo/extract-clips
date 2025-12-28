#!/usr/bin/env python3
"""
Motion Clip Extractor

Detects motion in Ring camera footage and extracts clips containing movement.
Ideal for monitoring pets in mostly-static video.
"""

import os
from datetime import datetime, timedelta

from src.constants import INPUT_DIR, OUTPUT_DIR, VIDEO_EXTENSIONS
from src.extraction import extract_clip
from src.motion_detection import detect_motion_timestamps, merge_timestamps_into_ranges
from src.utils import get_video_duration


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
