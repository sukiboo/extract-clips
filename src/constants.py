# videos directories
INPUT_DIR = "./videos/inputs"
OUTPUT_DIR = "./videos/outputs"

# motion detection settings
MOTION_THRESHOLD_PERCENT = 0.05  # Min contour area as % of frame
MIN_CLIP_DURATION = 5.0  # Seconds - ignore very brief motion
MERGE_GAP = 5.0  # Seconds - merge motion events closer than this
BUFFER_BEFORE = 2.0  # Seconds to include before motion starts
BUFFER_AFTER = 2.0  # Seconds to include after motion ends

# video processing settings
FRAME_SKIP = 10  # Process every Nth frame for speed
FALLBACK_FPS = 30.0  # FPS to use if video metadata is missing
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# background subtractor settings (MOG2 algorithm)
# https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
BG_HISTORY_FRAMES = 500  # Actual video frames used to build background model
BG_HISTORY = BG_HISTORY_FRAMES // FRAME_SKIP  # Adjusted for frame skipping
BG_VAR_THRESHOLD = 50  # Variance threshold for background/foreground
BG_DETECT_SHADOWS = False  # Whether to detect and mark shadows
