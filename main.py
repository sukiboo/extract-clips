from src.motion_detection import process_videos
from src.utils import list_video_files


def extract_clips():
    """Extract clips from video files."""
    video_files = list_video_files()
    process_videos(video_files)


if __name__ == "__main__":
    extract_clips()
