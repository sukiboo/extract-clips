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
    MOTION_THRESHOLD_PERCENT_MAX,
    MOTION_THRESHOLD_PERCENT_MIN,
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
    raw_ranges, motion_frames = detect_motion_ranges_with_progress(video_path, duration)

    ranges = merge_motion_ranges(raw_ranges, duration)
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


def detect_motion_ranges_with_progress(
    video_path: str, duration: float
) -> tuple[list[tuple[float, float]], int]:
    """Detect motion ranges using hysteresis thresholding, with progress bar.

    Uses two thresholds:
    - MIN: Start/extend potential clips when motion exceeds this
    - MAX: Confirm a clip should be saved when motion exceeds this

    This captures "bursty" motion patterns (like a cat running/jumping) along with
    the lead-up and wind-down, while ignoring slow continuous motion that never
    gets dramatic enough to exceed MAX.

    Args:
        video_path: Path to the video file to process.
        duration: Video duration in seconds.

    Returns:
        A tuple of (confirmed motion ranges, motion frame count).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = FALLBACK_FPS

    # Calculate motion thresholds from frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_width * frame_height
    threshold_min = frame_area * (MOTION_THRESHOLD_PERCENT_MIN / 100)
    threshold_max = frame_area * (MOTION_THRESHOLD_PERCENT_MAX / 100)

    # Background subtractor - good for static cameras
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESHOLD, detectShadows=BG_DETECT_SHADOWS
    )

    confirmed_ranges: list[tuple[float, float]] = []
    frame_idx = 0
    motion_frames = 0

    # State machine for hysteresis tracking
    tracking = False
    track_start = 0.0
    has_dramatic_motion = False

    with tqdm(
        total=int(duration),
        bar_format="  {desc}|{bar:50}| {percentage:3.0f}%",
        leave=False,
        ascii=" #",
    ) as pbar:
        pbar.set_description("[0 motion frames] ")

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

            # Hysteresis state machine
            if frame_max_area >= threshold_min:
                motion_frames += 1
                pbar.set_description(f"[{motion_frames} motion frames] ")

                if not tracking:
                    # Start tracking potential clip
                    tracking = True
                    track_start = current_time
                    has_dramatic_motion = False

                if frame_max_area >= threshold_max:
                    # Dramatic motion detected - confirm this clip
                    has_dramatic_motion = True

            elif tracking:
                # Motion dropped below MIN - end tracking
                if has_dramatic_motion:
                    # Save confirmed clip
                    confirmed_ranges.append((track_start, current_time))
                # Reset state
                tracking = False
                has_dramatic_motion = False

            frame_idx += 1

        # Handle case where video ends while tracking
        if tracking and has_dramatic_motion:
            confirmed_ranges.append((track_start, duration))

        # Final update
        pbar.n = int(duration)
        pbar.refresh()

    cap.release()
    return confirmed_ranges, motion_frames


def merge_motion_ranges(
    ranges: list[tuple[float, float]], video_duration: float
) -> list[tuple[float, float]]:
    """Merge nearby motion ranges, add buffers, and filter short clips.

    Args:
        ranges: List of (start, end) time ranges where motion occurs.
        video_duration: The duration of the video in seconds.

    Returns:
        A list of merged and buffered time ranges.
    """
    if not ranges:
        return []

    # Sort by start time
    sorted_ranges = sorted(ranges, key=lambda r: r[0])

    # Merge nearby ranges
    merged = []
    current_start, current_end = sorted_ranges[0]

    for start, end in sorted_ranges[1:]:
        if start - current_end <= MERGE_GAP:
            # Extend current range
            current_end = max(current_end, end)
        else:
            # Save current range and start new one
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    # Don't forget the last range
    merged.append((current_start, current_end))

    # Add buffers and filter short clips
    final_ranges = []
    for start, end in merged:
        duration = end - start
        if duration < MIN_CLIP_DURATION:
            continue

        # Add buffers, clamped to video bounds
        buffered_start = max(0, start - BUFFER_BEFORE)
        buffered_end = min(video_duration, end + BUFFER_AFTER)
        final_ranges.append((buffered_start, buffered_end))

    return final_ranges
