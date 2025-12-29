# Extract Motion Clips from Videos

Extract motion clips from security camera footage. Scans videos for movement and saves only the interesting parts.
I use it to scan lots of Ring recordings to extract the clips of cats messing around 😼

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
2. **Detects** motion using hysteresis thresholding (two thresholds)
3. **Merges** nearby motion events into continuous time ranges
4. **Extracts** clips with ffmpeg (no re-encoding, fast)

### Hysteresis Thresholding

Uses two thresholds to capture "bursty" motion (like a cat running/jumping) while ignoring slow, boring movement (like a cat slowly walking):

- **MAX threshold**: Triggers clip capture when dramatic motion is detected
- **MIN threshold**: Extends the clip while any motion continues

```
Motion:  ▁▂▃▂▁▁▂▅▇█▆▃▂▁▁
                ─MAX─      (dramatic motion triggers capture)
              ───MIN────   (extends clip while motion continues)
              ↑        ↑
            start     end
```

This captures the full context around interesting moments without triggering on slow, continuous motion that never gets dramatic.

## Configuration

Edit `src/constants.py` to tune behavior:

| Setting | Default | Description |
|---------|---------|-------------|
| `MOTION_THRESHOLD_PERCENT_MAX` | 0.25 | Dramatic motion threshold (% of frame) -- must exceed this to trigger clip |
| `MOTION_THRESHOLD_PERCENT_MIN` | 0.05 | Extend clip while motion exceeds this |
| `MIN_CLIP_DURATION` | 12.0s | Ignore clips shorter than this |
| `MERGE_GAP` | 12.0s | Merge events closer than this |
| `BUFFER_BEFORE` | 2.0s | Seconds to include before motion |
| `BUFFER_AFTER` | 4.0s | Seconds to include after motion |
| `FRAME_SKIP` | 4 | Process every Nth frame (speed vs precision) |

## Supported Formats

`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
