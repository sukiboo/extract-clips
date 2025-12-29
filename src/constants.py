# videos directories
INPUT_DIR = "./videos/inputs"
OUTPUT_DIR = "./videos/outputs"

# motion detection settings (hysteresis thresholding)
MOTION_THRESHOLD_PERCENT_MAX = 0.25  # Trigger clip capture when motion exceeds this
MOTION_THRESHOLD_PERCENT_MIN = 0.05  # Extend clip as long as motion exceeds this
MIN_CLIP_DURATION = 12.0  # Seconds - ignore very brief motion
MERGE_GAP = 12.0  # Seconds - merge motion events closer than this
BUFFER_BEFORE = 2.0  # Seconds to include before motion starts
BUFFER_AFTER = 4.0  # Seconds to include after motion ends

# video processing settings
FRAME_SKIP = 4  # Process every Nth frame for speed
FALLBACK_FPS = 30.0  # FPS to use if video metadata is missing
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# background subtractor settings (MOG2 algorithm)
# https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
BG_HISTORY_FRAMES = 300  # Actual video frames used to build background model
BG_HISTORY = BG_HISTORY_FRAMES // FRAME_SKIP  # Adjusted for frame skipping
BG_VAR_THRESHOLD = 100  # Variance threshold for background/foreground
BG_DETECT_SHADOWS = False  # Whether to detect and mark shadows
