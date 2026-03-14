#!/usr/bin/env python3
"""Music Genre Classifier – Command-Line Interface.

Usage
-----
    python app.py path/to/song.mp3
    python app.py path/to/song.wav --duration 60
    python app.py path/to/song.flac --top 5

The tool analyses the audio file, predicts the most likely genre, prints a
confidence score, and then recommends similar genres with representative
artists.
"""

import argparse
import sys

from music_classifier import MusicClassifier, MusicRecommender
from music_classifier.classifier import GENRE_DESCRIPTIONS


def _bar(value: float, width: int = 20) -> str:
    """Return a simple ASCII progress bar for a 0–1 value."""
    filled = round(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def run(audio_path: str, duration: float | None, top_n: int) -> None:
    """Classify *audio_path* and print genre + recommendations."""
    print(f"\n🎵  Analysing: {audio_path}")
    if duration:
        print(f"    (using first {duration:.0f} s of audio)\n")
    else:
        print("    (using full audio)\n")

    classifier = MusicClassifier()
    genre, confidence, scores = classifier.classify(audio_path, duration=duration)

    # --- Classification result ---
    print("=" * 56)
    print("  CLASSIFICATION RESULT")
    print("=" * 56)
    desc = GENRE_DESCRIPTIONS.get(genre, genre.title())
    print(f"  Genre      : {genre.upper()}")
    print(f"  Confidence : {_bar(confidence)} {confidence:.1%}")
    print(f"  About      : {desc}")
    print()

    # --- Top-5 scores table ---
    print("  Genre scores (all genres):")
    for g, s in list(scores.items())[:10]:
        marker = " ◀" if g == genre else "  "
        print(f"    {g:<12} {_bar(s, 15)} {s:.1%}{marker}")
    print()

    # --- Recommendations ---
    recommender = MusicRecommender()
    recommendations = recommender.recommend(genre, top_n=top_n)

    print("=" * 56)
    print("  SIMILAR GENRES YOU MIGHT ENJOY")
    print("=" * 56)
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['genre'].upper()}")
        print(f"     Why: {rec['reason']}")
        print(f"     Artists: {', '.join(rec['artists'])}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify music genre and get recommendations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio_path", help="Path to the audio file to classify")
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Seconds of audio to analyse (default: 30, 0 = full file)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        metavar="N",
        dest="top_n",
        help="Number of similar genres to recommend (default: 3)",
    )
    args = parser.parse_args(argv)
    duration = args.duration if args.duration > 0 else None

    try:
        run(args.audio_path, duration=duration, top_n=args.top_n)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
