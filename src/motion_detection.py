import cv2

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
)


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

        # Check if motion exceeds threshold
        if frame_max_area > motion_threshold:
            timestamp = frame_idx / fps
            motion_timestamps.append(timestamp)

        frame_idx += 1

    cap.release()
    return motion_timestamps


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
