import os
from datetime import timedelta

import cv2
from tqdm import tqdm

from src.constants import (
    BG_DETECT_SHADOWS,
    BG_HISTORY,
    BG_VAR_THRESHOLD,
    BUFFER_AFTER,
    BUFFER_BEFORE,
    FALLBACK_FPS,
    FRAME_SKIP,
    MERGE_GAP,
    MIN_CLIP_DURATION,
    MOTION_THRESHOLD_PERCENT,
    OUTPUT_DIR,
)
from src.utils import extract_clip, get_video_start_time


def process_videos(video_files: list[str]) -> None:
    """Process videos and print summary.

    Args:
        video_files: List of video file paths to process.
    """
    if not video_files:
        return

    total = len(video_files)
    total_clips = 0

    for i, video_path in enumerate(sorted(video_files), 1):
        clips = process_video(video_path, i, total)
        total_clips += clips

    print(f"\nExtracted {total_clips} clips to `{OUTPUT_DIR}`")


def process_video(video_path: str, index: int, total: int) -> int:
    """Process a single video: detect motion, merge timestamps, extract clips.

    Args:
        video_path: Path to the video file to process.
        index: Current video index (1-based).
        total: Total number of videos.

    Returns:
        The number of clips extracted.
    """
    video_name = os.path.basename(video_path)
    prefix = f"[{index}/{total}] {video_name}"

    duration = get_video_duration(video_path)
    if duration <= 0:
        print(f"{prefix} -- error: could not read video")
        return 0

    # Print header line (will be overwritten with summary later)
    print(prefix)

    # Detect motion with progress bar on next line
    timestamps, motion_frames = detect_motion_timestamps_with_progress(video_path, duration)

    ranges = merge_timestamps_into_ranges(timestamps, duration)
    clips_extracted = 0

    if ranges:
        video_start_time = get_video_start_time(video_path)

        for start, end in ranges:
            motion_time = video_start_time + timedelta(seconds=start)
            time_str = motion_time.strftime("%Y-%m-%d_%H.%M.%S")
            output_name = f"{time_str}.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_name)

            if extract_clip(video_path, output_path, start, end):
                clips_extracted += 1

    # Move cursor up one line, overwrite with summary
    print(f"\033[A\r{prefix} -- {clips_extracted} clips, {motion_frames} motion frames")

    return clips_extracted


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using OpenCV.

    Args:
        video_path: Path to the video file to process.

    Returns:
        The duration of the video in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0:
        return 0.0

    return frame_count / fps


def detect_motion_timestamps_with_progress(
    video_path: str, duration: float
) -> tuple[list[float], int]:
    """Detect timestamps where motion occurs, with progress bar.

    Args:
        video_path: Path to the video file to process.
        duration: Video duration in seconds.

    Returns:
        A tuple of (timestamps list, motion frame count).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = FALLBACK_FPS

    # Calculate motion threshold from frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_width * frame_height
    motion_threshold = frame_area * (MOTION_THRESHOLD_PERCENT / 100)

    # Background subtractor - good for static cameras
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESHOLD, detectShadows=BG_DETECT_SHADOWS
    )

    motion_timestamps = []
    frame_idx = 0
    motion_frames = 0

    with tqdm(
        total=int(duration),
        bar_format="  {desc}|{bar:50}| {percentage:3.0f}%",
        leave=False,
        ascii=" #",
    ) as pbar:
        pbar.set_description("[0 clips, 0 motion frames] ")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps

            # Update progress bar
            pbar.n = min(int(current_time), int(duration))
            pbar.refresh()

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

            # Check if motion exceeds threshold
            if frame_max_area > motion_threshold:
                timestamp = frame_idx / fps
                motion_timestamps.append(timestamp)
                motion_frames += 1
                # Update description with running count
                clips_so_far = len(merge_timestamps_into_ranges(motion_timestamps, duration))
                pbar.set_description(f"[{clips_so_far} clips, {motion_frames} motion frames] ")

            frame_idx += 1

        # Final update
        pbar.n = int(duration)
        pbar.refresh()

    cap.release()
    return motion_timestamps, motion_frames


def merge_timestamps_into_ranges(
    timestamps: list[float], video_duration: float
) -> list[tuple[float, float]]:
    """Merge nearby motion timestamps into continuous time ranges.

    Args:
        timestamps: List of timestamps where motion occurs.
        video_duration: The duration of the video in seconds.

    Returns:
        A list of time ranges where motion occurs.
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
