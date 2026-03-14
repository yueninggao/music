"""
main.py
=======
CLI entry-point for the Music-Video Recommendation System.

Usage
-----
    python main.py --source synthetic:60          # 60 s synthetic demo
    python main.py --source /path/to/video.mp4    # real video file (requires OpenCV)
    python main.py --source synthetic:45 --top-n 3 --json
    python main.py --source synthetic:90 --interval 2.0 --top-n 5
"""

from __future__ import annotations

import argparse
import json
import sys

from src.pipeline import MusicVideoPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="music",
        description="Music-Video Recommendation System: "
        "classifies a video and recommends music with a creative brief.",
    )
    p.add_argument(
        "--source",
        default="synthetic:60",
        help=(
            'Video source. Use "synthetic:<seconds>" for a demo '
            "(default: synthetic:60), or a path to a real video file."
        ),
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=5,
        metavar="N",
        help="Number of music recommendations to generate (default: 5).",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Frame sampling interval in seconds (default: 1.0).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=300,
        metavar="N",
        help="Maximum frames to sample from the video (default: 300).",
    )
    p.add_argument(
        "--opencv",
        action="store_true",
        help="Use OpenCV for real frame extraction (requires cv2 installed).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of the human-readable report.",
    )
    return p


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)

    pipeline = MusicVideoPipeline(
        sample_interval_sec=args.interval,
        max_frames=args.max_frames,
        top_n_recommendations=args.top_n,
        use_opencv=args.opencv,
    )

    result = pipeline.run(args.source)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        result.print_report()

    return 0


if __name__ == "__main__":
    sys.exit(main())
