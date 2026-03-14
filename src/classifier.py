"""
classifier.py
=============
VIDEO RECOGNITION & CLASSIFICATION stage of the pipeline.

Takes a FrameBundle (produced by VideoProcessor) and outputs a
ClassificationResult that describes the video in terms of:

  • Scene type  (e.g. outdoor, indoor, concert, nature, urban …)
  • Activity    (e.g. dancing, sports, relaxing, driving, celebration …)
  • Mood        (e.g. energetic, calm, melancholic, happy, tense …)
  • Pace        (slow / moderate / fast) derived from motion scores
  • Lighting    (dark / normal / bright) derived from brightness

DECISION-MAKING: The classifier uses rule-based thresholds to map
aggregated frame statistics to semantic labels.  A confidence score
is produced for every label.  Where a real deep-learning model is
available (e.g. a CLIP-based scene classifier), it can be wired in
via the optional `model_fn` injection point without changing the API.

Output: ClassificationResult dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

from .video_processor import FrameBundle


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class LabelScore:
    """A single classification label with a confidence value."""
    label: str
    confidence: float   # 0.0 – 1.0

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class ClassificationResult:
    """
    Full classification output for a video clip.

    Each field holds an ordered list of LabelScore objects (highest
    confidence first) so callers can easily access the top prediction
    or inspect alternatives.
    """
    source_hash: str

    scenes: List[LabelScore] = field(default_factory=list)
    activities: List[LabelScore] = field(default_factory=list)
    moods: List[LabelScore] = field(default_factory=list)
    pace: List[LabelScore] = field(default_factory=list)
    lighting: List[LabelScore] = field(default_factory=list)

    @property
    def top_scene(self) -> str:
        return self.scenes[0].label if self.scenes else "unknown"

    @property
    def top_activity(self) -> str:
        return self.activities[0].label if self.activities else "unknown"

    @property
    def top_mood(self) -> str:
        return self.moods[0].label if self.moods else "unknown"

    @property
    def top_pace(self) -> str:
        return self.pace[0].label if self.pace else "moderate"

    @property
    def top_lighting(self) -> str:
        return self.lighting[0].label if self.lighting else "normal"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_hash": self.source_hash,
            "scene": self.top_scene,
            "activity": self.top_activity,
            "mood": self.top_mood,
            "pace": self.top_pace,
            "lighting": self.top_lighting,
            "scene_scores": [{"label": s.label, "confidence": s.confidence} for s in self.scenes],
            "activity_scores": [{"label": a.label, "confidence": a.confidence} for a in self.activities],
            "mood_scores": [{"label": m.label, "confidence": m.confidence} for m in self.moods],
        }

    def __str__(self) -> str:
        return (
            f"[Classification] scene={self.top_scene} activity={self.top_activity} "
            f"mood={self.top_mood} pace={self.top_pace} lighting={self.top_lighting}"
        )


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

# Scene rules: (name, condition_fn)
#   condition_fn receives (avg_brightness, avg_motion, avg_edge, dom_color)
_SCENE_RULES: List[tuple] = [
    ("concert",  lambda b, m, e, c: m > 0.30 and e > 0.55),
    ("nature",   lambda b, m, e, c: b > 0.55 and c["G"] > 0.36 and m < 0.20),
    ("urban",    lambda b, m, e, c: e > 0.50 and b < 0.65),
    ("indoor",   lambda b, m, e, c: b < 0.45 and e < 0.45),
    ("beach",    lambda b, m, e, c: b > 0.60 and c["B"] > 0.36),
    ("night",    lambda b, m, e, c: b < 0.30),
    ("studio",   lambda b, m, e, c: 0.40 <= b <= 0.65 and e < 0.40),
    ("outdoor",  lambda b, m, e, c: b > 0.50),
]

_ACTIVITY_RULES: List[tuple] = [
    ("dancing",     lambda b, m, e: m > 0.35),
    ("sports",      lambda b, m, e: m > 0.25 and e > 0.45),
    ("celebration", lambda b, m, e: m > 0.20 and b > 0.55),
    ("driving",     lambda b, m, e: m > 0.15 and e > 0.40),
    ("relaxing",    lambda b, m, e: m < 0.10),
    ("performing",  lambda b, m, e: m > 0.10 and e > 0.35),
    ("travel",      lambda b, m, e: m > 0.12 and b > 0.45),
    ("ambient",     lambda b, m, e: True),   # catch-all
]

_MOOD_RULES: List[tuple] = [
    ("energetic",   lambda b, m, e: m > 0.30 and b > 0.50),
    ("happy",       lambda b, m, e: b > 0.60 and m > 0.10),
    ("romantic",    lambda b, m, e: b > 0.45 and m < 0.15 and e < 0.45),
    ("melancholic", lambda b, m, e: b < 0.35),
    ("tense",       lambda b, m, e: e > 0.60 and m > 0.20),
    ("calm",        lambda b, m, e: m < 0.12 and b > 0.40),
    ("mysterious",  lambda b, m, e: b < 0.45 and e > 0.50),
    ("neutral",     lambda b, m, e: True),   # catch-all
]

_PACE_RULES: List[tuple] = [
    ("fast",     lambda m: m > 0.28),
    ("moderate", lambda m: 0.10 <= m <= 0.28),
    ("slow",     lambda m: m < 0.10),
]

_LIGHTING_RULES: List[tuple] = [
    ("dark",   lambda b: b < 0.30),
    ("dim",    lambda b: 0.30 <= b < 0.45),
    ("normal", lambda b: 0.45 <= b <= 0.65),
    ("bright", lambda b: b > 0.65),
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class VideoClassifier:
    """
    Classifies a FrameBundle into semantic labels using a tiered strategy:

    1. If an external ``model_fn`` is provided, it is called first and its
       output merged with the rule-based scores (model gets 2× weight).
    2. Rule-based heuristics are applied to aggregate frame statistics.
    3. Results are sorted by confidence (descending) and returned as a
       ``ClassificationResult``.

    Parameters
    ----------
    model_fn : callable, optional
        A function ``(FrameBundle) -> dict`` that returns partial classification
        data (same keys as ``ClassificationResult.as_dict``).  Can be used to
        inject a real ML model without changing this class.
    min_confidence : float
        Labels with confidence below this threshold are excluded.
    """

    def __init__(
        self,
        model_fn: Optional[Callable[[FrameBundle], Dict[str, Any]]] = None,
        min_confidence: float = 0.05,
    ) -> None:
        self.model_fn = model_fn
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, bundle: FrameBundle) -> ClassificationResult:
        """
        Classify the video described by *bundle*.

        Returns
        -------
        ClassificationResult
        """
        b = bundle.avg_brightness
        m = bundle.avg_motion
        e = bundle.avg_edge_density
        c = bundle.dominant_color or {"R": 0.33, "G": 0.33, "B": 0.33}

        result = ClassificationResult(source_hash=bundle.content_hash)
        result.scenes = self._apply_rules(_SCENE_RULES, b, m, e, c, n_args=4)
        result.activities = self._apply_rules(_ACTIVITY_RULES, b, m, e, n_args=3)
        result.moods = self._apply_rules(_MOOD_RULES, b, m, e, n_args=3)
        result.pace = self._apply_pace(m)
        result.lighting = self._apply_lighting(b)

        # Optional model fusion
        if self.model_fn is not None:
            result = self._fuse_model(result, bundle)

        return result

    # ------------------------------------------------------------------
    # Rule application helpers
    # ------------------------------------------------------------------

    def _apply_rules(
        self,
        rules: List[tuple],
        b: float,
        m: float,
        e: float,
        c: Optional[Dict[str, float]] = None,
        n_args: int = 3,
    ) -> List[LabelScore]:
        scores: List[LabelScore] = []
        matched: bool = False

        for name, fn in rules:
            if n_args == 4:
                hit = fn(b, m, e, c)
            else:
                hit = fn(b, m, e)

            # Confidence: base 0.5 for a direct rule hit; scale by proximity of
            # triggering metrics to their boundary values.
            if hit:
                conf = self._confidence_from_hit(name, b, m, e)
                if conf >= self.min_confidence:
                    scores.append(LabelScore(label=name, confidence=conf))
                matched = True

        # If nothing matched, add catch-all with low confidence
        if not matched:
            scores.append(LabelScore(label="unknown", confidence=0.1))

        # Sort descending by confidence and deduplicate
        scores.sort(key=lambda s: s.confidence, reverse=True)
        return scores

    @staticmethod
    def _confidence_from_hit(name: str, b: float, m: float, e: float) -> float:
        """
        Assign a confidence value (0–1) for a matched rule based on how
        strongly the triggering metrics support the label.
        """
        # Motion-based labels
        if name in ("energetic", "dancing", "sports", "fast"):
            return min(1.0, 0.5 + m * 1.2)
        # Brightness-based labels
        if name in ("happy", "bright", "beach"):
            return min(1.0, 0.5 + b * 0.7)
        # Dark / melancholic
        if name in ("melancholic", "dark", "night"):
            return min(1.0, 0.5 + (1.0 - b) * 0.8)
        # Edge-based
        if name in ("tense", "urban", "concert"):
            return min(1.0, 0.5 + e * 0.7)
        # Calm / relaxing
        if name in ("calm", "relaxing", "slow"):
            return min(1.0, 0.5 + (1.0 - m) * 0.6)
        # Catch-all labels
        if name in ("ambient", "neutral", "outdoor", "studio"):
            return 0.40
        return 0.50

    def _apply_pace(self, m: float) -> List[LabelScore]:
        scores = []
        for name, fn in _PACE_RULES:
            if fn(m):
                if name == "fast":
                    scores.append(LabelScore(label=name, confidence=min(1.0, 0.5 + m * 1.5)))
                elif name == "slow":
                    scores.append(LabelScore(label=name, confidence=min(1.0, 0.5 + (1 - m) * 1.2)))
                else:
                    scores.append(LabelScore(label=name, confidence=0.65))
        scores.sort(key=lambda s: s.confidence, reverse=True)
        return scores or [LabelScore(label="moderate", confidence=0.5)]

    def _apply_lighting(self, b: float) -> List[LabelScore]:
        scores = []
        for name, fn in _LIGHTING_RULES:
            if fn(b):
                conf = b if name == "bright" else (1.0 - b) if name == "dark" else 0.6
                scores.append(LabelScore(label=name, confidence=min(1.0, 0.4 + conf * 0.6)))
        scores.sort(key=lambda s: s.confidence, reverse=True)
        return scores or [LabelScore(label="normal", confidence=0.5)]

    # ------------------------------------------------------------------
    # Model fusion
    # ------------------------------------------------------------------

    def _fuse_model(
        self, result: ClassificationResult, bundle: FrameBundle
    ) -> ClassificationResult:
        """
        Merge external model predictions (2× weight) with rule-based scores.
        Model output is expected as a dict like:
          {"scene": "concert", "mood": "energetic", ...}
        """
        try:
            model_out: Dict[str, Any] = self.model_fn(bundle)  # type: ignore[misc]
        except Exception:
            return result  # fall back to rule-based on model failure

        def _boost(scores: List[LabelScore], top_label: str) -> List[LabelScore]:
            found = False
            for ls in scores:
                if ls.label == top_label:
                    ls.confidence = min(1.0, ls.confidence * 2.0)
                    found = True
            if not found:
                scores.append(LabelScore(label=top_label, confidence=0.70))
            scores.sort(key=lambda s: s.confidence, reverse=True)
            return scores

        if "scene" in model_out:
            result.scenes = _boost(result.scenes, model_out["scene"])
        if "activity" in model_out:
            result.activities = _boost(result.activities, model_out["activity"])
        if "mood" in model_out:
            result.moods = _boost(result.moods, model_out["mood"])
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def summarise(cr: ClassificationResult) -> str:
        return str(cr)
