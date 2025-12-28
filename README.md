# Extract Motion Clips from Videos

Extract motion clips from security camera footage. Scans videos for movement and saves only the interesting parts.

## Quick Start

```bash
pip install -r requirements.txt
```

Place videos in `./videos/inputs/`, then run:

```bash
python main.py
```

Clips are saved to `./videos/outputs/`.

## How It Works

1. **Scans** each video frame-by-frame using OpenCV background subtraction
2. **Detects** motion by finding contours that exceed a threshold
3. **Merges** nearby motion events into continuous time ranges
4. **Extracts** clips with ffmpeg (no re-encoding, fast)

## Configuration

Edit `src/constants.py` to tune behavior:

| Setting | Default | Description |
|---------|---------|-------------|
| `MOTION_THRESHOLD_PERCENT` | 0.05 | Motion sensitivity (% of frame area) |
| `MIN_CLIP_DURATION` | 5.0s | Ignore motion shorter than this |
| `MERGE_GAP` | 5.0s | Merge events closer than this |
| `BUFFER_BEFORE` | 2.0s | Seconds to include before motion |
| `BUFFER_AFTER` | 2.0s | Seconds to include after motion |
| `FRAME_SKIP` | 10 | Process every Nth frame (speed vs precision) |

## Supported Formats

`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
