"""
video_processor.py
==================
Handles all INPUT stages of the music-video pipeline:
  1. Load a video file (or a URL / webcam stream placeholder)
  2. Extract representative frames at a configurable interval
  3. Compute per-frame visual features (brightness, colour histogram, motion estimate)
  4. Aggregate features across the full clip into a single FrameBundle

All heavy ML / CV dependencies (e.g. OpenCV, PIL) are treated as optional so the
module works in environments where they are not installed (using a lightweight
pure-Python fallback).  Real deployments should install the optional packages.
"""

from __future__ import annotations

import os
import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FrameFeatures:
    """Low-level features extracted from a single video frame."""
    frame_index: int
    timestamp_sec: float
    brightness: float          # 0.0–1.0
    color_histogram: Dict[str, float]   # {"R": ..., "G": ..., "B": ...} – normalised
    edge_density: float        # 0.0–1.0  (proxy for visual complexity)
    motion_score: float        # 0.0–1.0  (optical-flow magnitude vs. previous frame)


@dataclass
class FrameBundle:
    """Aggregated representation of an entire video clip."""
    source_path: str
    total_frames_sampled: int
    duration_sec: float
    frames: List[FrameFeatures] = field(default_factory=list)

    # Aggregated statistics derived from individual frames
    avg_brightness: float = 0.0
    avg_motion: float = 0.0
    avg_edge_density: float = 0.0
    dominant_color: Dict[str, float] = field(default_factory=dict)
    content_hash: str = ""      # SHA-256 of source path + frame count (for caching)

    def compute_aggregates(self) -> None:
        """Compute aggregate statistics from stored frames."""
        if not self.frames:
            return
        n = len(self.frames)
        self.avg_brightness = sum(f.brightness for f in self.frames) / n
        self.avg_motion = sum(f.motion_score for f in self.frames) / n
        self.avg_edge_density = sum(f.edge_density for f in self.frames) / n
        self.dominant_color = {
            ch: sum(f.color_histogram.get(ch, 0.0) for f in self.frames) / n
            for ch in ("R", "G", "B")
        }
        raw = f"{self.source_path}:{self.total_frames_sampled}"
        self.content_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Core processor
# ---------------------------------------------------------------------------

class VideoProcessor:
    """
    Extracts frames and per-frame features from a video file.

    Parameters
    ----------
    sample_interval_sec : float
        How often (in seconds) to sample a frame.  Default is 1 s.
    max_frames : int
        Hard cap on the number of frames to process.
    use_opencv : bool
        If True (and OpenCV is installed) use real frame reading.
        If False (default) a deterministic synthetic generator is used –
        useful for testing without a real video file.
    """

    def __init__(
        self,
        sample_interval_sec: float = 1.0,
        max_frames: int = 300,
        use_opencv: bool = False,
    ) -> None:
        if sample_interval_sec <= 0:
            raise ValueError("sample_interval_sec must be positive")
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        self.sample_interval_sec = sample_interval_sec
        self.max_frames = max_frames
        self.use_opencv = use_opencv

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, source: str) -> FrameBundle:
        """
        Main entry-point.  Accepts a file path, directory of images, or a
        special ``"synthetic:<duration_sec>"`` URI for testing.

        Returns
        -------
        FrameBundle
            Populated bundle with per-frame features and aggregate statistics.
        """
        source = source.strip()
        if source.startswith("synthetic:"):
            return self._process_synthetic(source)
        if self.use_opencv:
            return self._process_opencv(source)
        # Fallback: treat source path as an identifier and generate synthetic data
        # derived from the path hash so results are deterministic.
        return self._process_synthetic(f"synthetic:30:{source}")

    # ------------------------------------------------------------------
    # Synthetic (test-friendly) backend
    # ------------------------------------------------------------------

    def _process_synthetic(self, uri: str) -> FrameBundle:
        """
        Generate deterministic synthetic frame data from a URI of the form:
            ``synthetic:<duration_sec>[:<seed_string>]``
        """
        parts = uri.split(":", 2)
        try:
            duration = float(parts[1]) if len(parts) > 1 else 30.0
        except ValueError:
            duration = 30.0
        seed_str = parts[2] if len(parts) > 2 else "default"

        frames = self._generate_synthetic_frames(duration, seed_str)
        bundle = FrameBundle(
            source_path=uri,
            total_frames_sampled=len(frames),
            duration_sec=duration,
            frames=frames,
        )
        bundle.compute_aggregates()
        return bundle

    def _generate_synthetic_frames(
        self, duration: float, seed_str: str
    ) -> List[FrameFeatures]:
        """Create deterministic pseudo-random frames based on ``seed_str``."""
        seed_hash = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16)
        rng_state = seed_hash

        def _next(lo: float = 0.0, hi: float = 1.0) -> float:
            nonlocal rng_state
            rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & (
                2**64 - 1
            )
            return lo + (rng_state / (2**64)) * (hi - lo)

        frames: List[FrameFeatures] = []
        t = 0.0
        idx = 0
        prev_brightness = _next(0.3, 0.9)

        while t < duration and idx < self.max_frames:
            brightness = max(0.0, min(1.0, prev_brightness + _next(-0.15, 0.15)))
            r, g, b = _next(), _next(), _next()
            total = r + g + b or 1.0
            motion = _next(0.0, 0.6) if idx > 0 else 0.0
            edge = _next(0.1, 0.9)

            frames.append(
                FrameFeatures(
                    frame_index=idx,
                    timestamp_sec=round(t, 3),
                    brightness=round(brightness, 4),
                    color_histogram={
                        "R": round(r / total, 4),
                        "G": round(g / total, 4),
                        "B": round(b / total, 4),
                    },
                    edge_density=round(edge, 4),
                    motion_score=round(motion, 4),
                )
            )
            prev_brightness = brightness
            t += self.sample_interval_sec
            idx += 1

        return frames

    # ------------------------------------------------------------------
    # OpenCV backend (optional)
    # ------------------------------------------------------------------

    def _process_opencv(self, path: str) -> FrameBundle:  # pragma: no cover
        """
        Real frame extraction using OpenCV.
        Only called when ``use_opencv=True`` and ``cv2`` is importable.
        """
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenCV (cv2) is required for real video processing. "
                "Install it with: pip install opencv-python"
            ) from exc

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_cv_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_cv_frames / fps
        step = max(1, int(fps * self.sample_interval_sec))

        frames: List[FrameFeatures] = []
        prev_gray: Optional[Any] = None
        cv_idx = 0

        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if cv_idx % step == 0:
                features = self._extract_cv_features(frame, prev_gray, len(frames))
                features.timestamp_sec = round(cv_idx / fps, 3)
                frames.append(features)
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv_idx += 1

        cap.release()
        bundle = FrameBundle(
            source_path=path,
            total_frames_sampled=len(frames),
            duration_sec=duration,
            frames=frames,
        )
        bundle.compute_aggregates()
        return bundle

    @staticmethod
    def _extract_cv_features(
        frame: Any, prev_gray: Optional[Any], idx: int
    ) -> FrameFeatures:  # pragma: no cover
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray)) / 255.0

        # Colour histogram (normalised per-channel mean)
        channels = cv2.split(frame)  # BGR
        total_px = frame.shape[0] * frame.shape[1] or 1
        b_mean = float(np.mean(channels[0])) / 255.0
        g_mean = float(np.mean(channels[1])) / 255.0
        r_mean = float(np.mean(channels[2])) / 255.0
        ch_sum = r_mean + g_mean + b_mean or 1.0

        # Edge density via Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / total_px

        # Motion via absolute frame difference
        motion = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion = float(np.mean(diff)) / 255.0

        return FrameFeatures(
            frame_index=idx,
            timestamp_sec=0.0,  # filled in by caller
            brightness=round(brightness, 4),
            color_histogram={
                "R": round(r_mean / ch_sum, 4),
                "G": round(g_mean / ch_sum, 4),
                "B": round(b_mean / ch_sum, 4),
            },
            edge_density=round(edge_density, 4),
            motion_score=round(motion, 4),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def summarise(bundle: FrameBundle) -> str:
        """Return a human-readable one-line summary of a FrameBundle."""
        return (
            f"[VideoProcessor] source='{bundle.source_path}' "
            f"frames={bundle.total_frames_sampled} "
            f"duration={bundle.duration_sec:.1f}s "
            f"avg_brightness={bundle.avg_brightness:.3f} "
            f"avg_motion={bundle.avg_motion:.3f} "
            f"hash={bundle.content_hash}"
        )
