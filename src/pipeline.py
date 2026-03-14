"""
pipeline.py
===========
END-TO-END ORCHESTRATION of the music-video recommendation system.

Connects all four stages in sequence:

    VideoProcessor  →  VideoClassifier  →  MusicRecommender  →  CreativePipeline

                       INPUT          CLASSIFY        RECOMMEND         CREATIVE OUTPUT

Exposes a single high-level class ``MusicVideoPipeline`` whose ``run()``
method accepts a video source string and returns a fully populated
``PipelineResult`` that bundles every intermediate and final output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List

from .video_processor import VideoProcessor, FrameBundle
from .classifier import VideoClassifier, ClassificationResult
from .recommender import MusicRecommender, RecommendationOutput, Track
from .creative_pipeline import CreativePipeline, CreativePackage


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Bundles every stage's output into one convenient object.

    Attributes
    ----------
    source          : the original input source string
    bundle          : frame-level features from VideoProcessor
    classification  : semantic labels from VideoClassifier
    recommendations : ranked music list from MusicRecommender
    creative        : final creative package from CreativePipeline
    elapsed_sec     : wall-clock time for the full pipeline run
    """
    source: str
    bundle: Optional[FrameBundle] = None
    classification: Optional[ClassificationResult] = None
    recommendations: Optional[RecommendationOutput] = None
    creative: Optional[CreativePackage] = None
    elapsed_sec: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "elapsed_sec": round(self.elapsed_sec, 4),
            "bundle": {
                "source_path": self.bundle.source_path,
                "total_frames": self.bundle.total_frames_sampled,
                "duration_sec": self.bundle.duration_sec,
                "avg_brightness": round(self.bundle.avg_brightness, 4),
                "avg_motion": round(self.bundle.avg_motion, 4),
                "content_hash": self.bundle.content_hash,
            } if self.bundle else {},
            "classification": self.classification.as_dict() if self.classification else {},
            "recommendations": self.recommendations.as_dict() if self.recommendations else {},
            "creative": self.creative.as_dict() if self.creative else {},
        }

    def print_report(self) -> None:
        """Print a structured human-readable report to stdout."""
        sep = "─" * 70
        print(sep)
        print(f"  Music-Video Pipeline Report")
        print(f"  Source  : {self.source}")
        print(f"  Runtime : {self.elapsed_sec:.3f}s")
        print(sep)

        if self.bundle:
            print("\n▸ INPUT / VIDEO PROCESSING")
            print(f"  Frames sampled : {self.bundle.total_frames_sampled}")
            print(f"  Duration       : {self.bundle.duration_sec:.1f}s")
            print(f"  Avg brightness : {self.bundle.avg_brightness:.3f}")
            print(f"  Avg motion     : {self.bundle.avg_motion:.3f}")
            print(f"  Avg edge density: {self.bundle.avg_edge_density:.3f}")
            print(f"  Content hash   : {self.bundle.content_hash}")

        if self.classification:
            print("\n▸ VIDEO RECOGNITION & CLASSIFICATION")
            print(f"  Scene      : {self.classification.top_scene}")
            print(f"  Activity   : {self.classification.top_activity}")
            print(f"  Mood       : {self.classification.top_mood}")
            print(f"  Pace       : {self.classification.top_pace}")
            print(f"  Lighting   : {self.classification.top_lighting}")

        if self.recommendations:
            print("\n▸ MUSIC RECOMMENDATIONS (decision output)")
            for i, r in enumerate(self.recommendations.recommendations, 1):
                print(f"  {i}. {r}")

        if self.creative:
            print("\n" + str(self.creative))

        print(sep)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class MusicVideoPipeline:
    """
    Orchestrates the full processing flow from raw video input to a
    rich creative music package.

    Parameters
    ----------
    sample_interval_sec : float
        Frame sampling interval for VideoProcessor.
    max_frames : int
        Maximum frames to sample from the video.
    top_n_recommendations : int
        Number of music recommendations to generate.
    use_opencv : bool
        Use real OpenCV frame reading (requires cv2 installed).
    model_fn : callable, optional
        External ML model for classification fusion.
    custom_catalogue : list of Track, optional
        Override the built-in music catalogue.
    """

    def __init__(
        self,
        sample_interval_sec: float = 1.0,
        max_frames: int = 300,
        top_n_recommendations: int = 5,
        use_opencv: bool = False,
        model_fn: Optional[Callable[[FrameBundle], Dict[str, Any]]] = None,
        custom_catalogue: Optional[List[Track]] = None,
    ) -> None:
        self._processor = VideoProcessor(
            sample_interval_sec=sample_interval_sec,
            max_frames=max_frames,
            use_opencv=use_opencv,
        )
        self._classifier = VideoClassifier(model_fn=model_fn)
        self._recommender = MusicRecommender(
            catalogue=custom_catalogue,
            top_n=top_n_recommendations,
        )
        self._top_n = top_n_recommendations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, source: str) -> PipelineResult:
        """
        Execute the full pipeline for *source*.

        Parameters
        ----------
        source : str
            Path to a video file, ``"synthetic:<dur_sec>"`` for testing,
            or any string identifier (will use deterministic synthetic data).

        Returns
        -------
        PipelineResult
        """
        t_start = time.monotonic()
        result = PipelineResult(source=source)

        # ── Stage 1: INPUT – video processing ─────────────────────────
        result.bundle = self._processor.process(source)

        # ── Stage 2: RECOGNITION & CLASSIFICATION ─────────────────────
        result.classification = self._classifier.classify(result.bundle)

        # ── Stage 3: DECISION-MAKING & RECOMMENDATION ─────────────────
        result.recommendations = self._recommender.recommend(result.classification)

        # ── Stage 4: CREATIVE STEPS ───────────────────────────────────
        creative = CreativePipeline(
            top_n_playlist=self._top_n,
            duration_sec=result.bundle.duration_sec,
        )
        result.creative = creative.run(
            result.recommendations,
            top_mood=result.classification.top_mood,
        )

        result.elapsed_sec = time.monotonic() - t_start
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def quick_run(source: str, top_n: int = 5) -> PipelineResult:
        """
        Class-level helper: instantiate a default pipeline and run it.
        Useful for one-liners in notebooks / scripts.
        """
        return MusicVideoPipeline(top_n_recommendations=top_n).run(source)
