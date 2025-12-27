# extract-clips

Detect motion and extract clips from recorded video. Ideal for extracting interesting moments from security camera footage (like monitoring pets).

## Requirements

- Python 3.10+

## Installation

```bash
pip install -r requirements.txt
```

This installs OpenCV and a bundled ffmpeg binary (no system install needed).

## Usage

1. Place your video files in `videos/inputs/`
2. Run the script:
   ```bash
   python extract_clips.py
   ```
3. Find extracted clips in `videos/outputs/`

Supported formats: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`

## Configuration

Edit the constants at the top of `extract_clips.py` to tune behavior:

| Setting | Default | Description |
|---------|---------|-------------|
| `MOTION_THRESHOLD` | 5000 | Min contour area to count as motion (lower = more sensitive) |
| `MIN_CLIP_DURATION` | 1.0s | Ignore motion events shorter than this |
| `MERGE_GAP` | 2.0s | Merge motion events closer than this into one clip |
| `BUFFER_BEFORE` | 1.0s | Include this much video before motion starts |
| `BUFFER_AFTER` | 1.0s | Include this much video after motion ends |
| `FRAME_SKIP` | 3 | Process every Nth frame (higher = faster but less precise) |

## How It Works

1. **Motion Detection**: Uses OpenCV's background subtraction (MOG2) to identify frames with movement
2. **Timestamp Merging**: Groups nearby motion events into continuous time ranges
3. **Clip Extraction**: Uses ffmpeg to extract clips without re-encoding (fast)
