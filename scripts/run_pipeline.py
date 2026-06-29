"""
Run the full player tracking pipeline on sports videos.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --video path/to/video.mkv
    python scripts/run_pipeline.py --max-frames 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import get_video_paths, load_config
from src.pipeline.tracking_pipeline import PlayerTrackingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track players in sports videos.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Process a single video file instead of all videos in video_dir",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames per video (useful for quick tests)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.max_frames is not None:
        config["processing"]["max_frames"] = args.max_frames

    pipeline = PlayerTrackingPipeline(config)

    if args.video:
        video_paths = [Path(args.video)]
    else:
        video_paths = get_video_paths(config)
        if not video_paths:
            print(f"No videos found in: {config['paths']['video_dir']}")
            sys.exit(1)

    print(f"Found {len(video_paths)} video(s) to process.")
    pipeline.process_all_videos(video_paths)
    print(f"\nDone. Results saved to: {config['paths']['output_dir']}")


if __name__ == "__main__":
    main()
