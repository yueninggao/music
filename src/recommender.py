"""
recommender.py
==============
DECISION-MAKING & RECOMMENDATION stage of the pipeline.

Takes a ClassificationResult and returns a ranked list of MusicRecommendation
objects.  Each recommendation includes:

  • Track title / artist / genre / sub-genre
  • BPM range  (matches video pace)
  • Energy level (matches video mood / activity)
  • Mood tags
  • Confidence score
  • Reason string (explainability)

DECISION LOGIC
--------------
A multi-attribute decision matrix maps classification dimensions to music
attributes.  The full library of candidate tracks is scored against the
video's requirements; the top-N candidates are returned.

The music library is built in (catalogue.py-style constant) and can be
replaced / extended with a real database or API call via the optional
``library`` injection point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .classifier import ClassificationResult


# ---------------------------------------------------------------------------
# Music catalogue
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """Represents a single track in the music catalogue."""
    id: str
    title: str
    artist: str
    genre: str
    subgenre: str
    bpm: int
    energy: float           # 0.0 – 1.0
    valence: float          # 0.0 – 1.0   (musical positivity)
    mood_tags: List[str] = field(default_factory=list)
    scene_tags: List[str] = field(default_factory=list)
    activity_tags: List[str] = field(default_factory=list)


# Built-in catalogue – a curated cross-genre sample
DEFAULT_CATALOGUE: List[Track] = [
    Track("t001", "Electric Pulse",      "Synthwave Project",  "Electronic", "Synthwave",    128, 0.90, 0.80,
          ["energetic", "tense"],         ["urban", "concert"], ["dancing", "performing"]),
    Track("t002", "Ocean Breeze",        "Chill Collective",   "Ambient",    "Chillout",      72, 0.25, 0.75,
          ["calm", "romantic"],           ["beach", "outdoor"], ["relaxing"]),
    Track("t003", "Midnight Streets",   "Lo-Fi Hours",         "Hip-Hop",    "Lo-Fi",         85, 0.35, 0.40,
          ["melancholic", "mysterious"], ["urban", "night"],   ["ambient"]),
    Track("t004", "Sunrise Run",        "Indie Pulse",         "Indie",      "Indie Pop",    110, 0.70, 0.85,
          ["happy", "energetic"],        ["outdoor", "nature"],["sports", "travel"]),
    Track("t005", "Festival Ground",    "EDM All Stars",       "Electronic", "House",        128, 0.95, 0.90,
          ["energetic", "happy"],        ["concert", "outdoor"],["dancing", "celebration"]),
    Track("t006", "Quiet Forest",       "Acoustic Wanderer",   "Folk",       "Acoustic Folk", 68, 0.20, 0.65,
          ["calm", "melancholic"],       ["nature", "outdoor"], ["relaxing"]),
    Track("t007", "City Lights",        "Urban Soul",          "R&B",        "Soul",          96, 0.60, 0.70,
          ["romantic", "calm"],          ["urban", "indoor"],  ["performing", "driving"]),
    Track("t008", "Neon Chase",         "Action Beats",        "Electronic", "Drum & Bass", 174, 0.92, 0.65,
          ["tense", "energetic"],        ["urban", "night"],   ["sports", "driving"]),
    Track("t009", "Golden Hour",        "Indie Vocals",        "Indie",      "Dream Pop",     88, 0.45, 0.80,
          ["happy", "romantic"],         ["outdoor", "beach"], ["relaxing", "travel"]),
    Track("t010", "Deep Space",         "Ambient Cosmos",      "Ambient",    "Space Ambient", 60, 0.15, 0.50,
          ["mysterious", "calm"],        ["studio", "indoor"], ["ambient"]),
    Track("t011", "Stadium Anthem",     "Rock United",         "Rock",       "Arena Rock",   140, 0.88, 0.85,
          ["energetic", "happy"],        ["concert", "outdoor"],["sports", "celebration"]),
    Track("t012", "Rainy Sunday",       "Jazz Trio",           "Jazz",       "Cool Jazz",     78, 0.30, 0.55,
          ["melancholic", "calm"],       ["indoor", "studio"], ["relaxing"]),
    Track("t013", "Summer Carnival",    "Pop Fusion",          "Pop",        "Dance Pop",    120, 0.80, 0.92,
          ["happy", "energetic"],        ["outdoor", "beach"], ["dancing", "celebration"]),
    Track("t014", "Slow Burn",          "Neo Soul",            "R&B",        "Neo Soul",      72, 0.40, 0.60,
          ["romantic", "calm"],          ["indoor", "studio"], ["performing"]),
    Track("t015", "Mountain Echo",      "Cinematic Score",     "Classical",  "Orchestral",    80, 0.55, 0.70,
          ["calm", "mysterious"],        ["nature", "outdoor"], ["travel", "ambient"]),
    Track("t016", "Trap Hustle",        "Street Beats",        "Hip-Hop",    "Trap",         140, 0.85, 0.55,
          ["energetic", "tense"],        ["urban"],             ["driving", "sports"]),
    Track("t017", "Piano Reverie",      "Solo Keys",           "Classical",  "Neoclassical",  52, 0.22, 0.68,
          ["melancholic", "romantic"],   ["indoor", "studio"],  ["relaxing", "ambient"]),
    Track("t018", "Afrobeats Groove",   "Pan African Sound",   "World",      "Afrobeats",    105, 0.75, 0.90,
          ["happy", "energetic"],        ["outdoor"],           ["dancing", "celebration"]),
    Track("t019", "Techno Underground", "Dark Matter",         "Electronic", "Techno",       145, 0.90, 0.45,
          ["tense", "energetic"],        ["concert", "urban"],  ["dancing", "performing"]),
    Track("t020", "Bossa Romance",      "Rio Strings",         "World",      "Bossa Nova",    88, 0.35, 0.72,
          ["romantic", "calm"],          ["beach", "indoor"],   ["relaxing"]),
]


# ---------------------------------------------------------------------------
# Recommendation result
# ---------------------------------------------------------------------------

@dataclass
class MusicRecommendation:
    """A single recommended track with explainability metadata."""
    track: Track
    score: float        # 0.0 – 1.0 (higher = better match)
    reasons: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        reasons_str = "; ".join(self.reasons[:3])
        return (
            f"[{self.score:.2f}] {self.track.title} – {self.track.artist} "
            f"({self.track.genre} / {self.track.subgenre}, {self.track.bpm} BPM) | {reasons_str}"
        )


@dataclass
class RecommendationOutput:
    """Container for all recommendations produced for a video."""
    source_hash: str
    classification_summary: str
    recommendations: List[MusicRecommendation] = field(default_factory=list)

    @property
    def top(self) -> Optional[MusicRecommendation]:
        return self.recommendations[0] if self.recommendations else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_hash": self.source_hash,
            "classification": self.classification_summary,
            "recommendations": [
                {
                    "rank": i + 1,
                    "title": r.track.title,
                    "artist": r.track.artist,
                    "genre": r.track.genre,
                    "subgenre": r.track.subgenre,
                    "bpm": r.track.bpm,
                    "energy": r.track.energy,
                    "score": round(r.score, 4),
                    "reasons": r.reasons,
                }
                for i, r in enumerate(self.recommendations)
            ],
        }


# ---------------------------------------------------------------------------
# BPM range mapping
# ---------------------------------------------------------------------------

_BPM_RANGES: Dict[str, tuple] = {
    "slow":     (50,  95),
    "moderate": (85, 125),
    "fast":     (115, 200),
}


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class MusicRecommender:
    """
    Scores every track in the catalogue against the video's ClassificationResult
    using a weighted multi-attribute scoring matrix, then returns the top-N
    recommendations.

    Scoring weights
    ---------------
    • mood match     – 0.35
    • activity match – 0.25
    • scene match    – 0.20
    • pace / BPM     – 0.15
    • energy match   – 0.05

    Parameters
    ----------
    catalogue : list of Track, optional
        Override the built-in catalogue with a custom library.
    top_n : int
        Maximum number of recommendations to return.
    """

    WEIGHTS = {
        "mood":     0.35,
        "activity": 0.25,
        "scene":    0.20,
        "pace":     0.15,
        "energy":   0.05,
    }

    def __init__(
        self,
        catalogue: Optional[List[Track]] = None,
        top_n: int = 5,
    ) -> None:
        self.catalogue = catalogue if catalogue is not None else DEFAULT_CATALOGUE
        self.top_n = top_n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, cr: ClassificationResult) -> RecommendationOutput:
        """
        Score every track against *cr* and return the top-N recommendations.
        """
        scored: List[MusicRecommendation] = []

        for track in self.catalogue:
            score, reasons = self._score_track(track, cr)
            scored.append(MusicRecommendation(track=track, score=score, reasons=reasons))

        scored.sort(key=lambda r: r.score, reverse=True)
        top = scored[: self.top_n]

        return RecommendationOutput(
            source_hash=cr.source_hash,
            classification_summary=str(cr),
            recommendations=top,
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_track(
        self, track: Track, cr: ClassificationResult
    ) -> tuple:
        """Return (score, reasons) for one track given a classification."""
        reasons: List[str] = []
        total = 0.0

        # ── Mood ──────────────────────────────────────────────────────
        mood_score, mood_reason = self._tag_match(
            [ls.label for ls in cr.moods],
            track.mood_tags,
            [ls.confidence for ls in cr.moods],
        )
        total += mood_score * self.WEIGHTS["mood"]
        if mood_reason:
            reasons.append(f"mood: {mood_reason}")

        # ── Activity ──────────────────────────────────────────────────
        act_score, act_reason = self._tag_match(
            [ls.label for ls in cr.activities],
            track.activity_tags,
            [ls.confidence for ls in cr.activities],
        )
        total += act_score * self.WEIGHTS["activity"]
        if act_reason:
            reasons.append(f"activity: {act_reason}")

        # ── Scene ─────────────────────────────────────────────────────
        scene_score, scene_reason = self._tag_match(
            [ls.label for ls in cr.scenes],
            track.scene_tags,
            [ls.confidence for ls in cr.scenes],
        )
        total += scene_score * self.WEIGHTS["scene"]
        if scene_reason:
            reasons.append(f"scene: {scene_reason}")

        # ── Pace / BPM ────────────────────────────────────────────────
        bpm_score, bpm_reason = self._bpm_match(track.bpm, cr.top_pace)
        total += bpm_score * self.WEIGHTS["pace"]
        if bpm_reason:
            reasons.append(bpm_reason)

        # ── Energy ────────────────────────────────────────────────────
        target_energy = self._energy_from_pace(cr.top_pace, cr.top_mood)
        energy_score = 1.0 - abs(track.energy - target_energy)
        total += energy_score * self.WEIGHTS["energy"]

        return round(total, 4), reasons

    @staticmethod
    def _tag_match(
        video_labels: List[str],
        track_tags: List[str],
        confidences: Optional[List[float]] = None,
    ) -> tuple:
        """
        Score how well a track's tags match the video's labels.
        Returns (score 0–1, best matching label or empty string).
        """
        if not video_labels or not track_tags:
            return 0.0, ""

        best_score = 0.0
        best_label = ""
        for i, label in enumerate(video_labels):
            if label in track_tags:
                conf = confidences[i] if confidences and i < len(confidences) else 0.5
                # Weight earlier (higher-ranked) labels more
                rank_weight = 1.0 / (i + 1)
                s = conf * rank_weight
                if s > best_score:
                    best_score = s
                    best_label = label

        # Normalise so top-1 perfect match → 1.0
        return min(1.0, best_score * 1.5), best_label

    @staticmethod
    def _bpm_match(bpm: int, pace: str) -> tuple:
        lo, hi = _BPM_RANGES.get(pace, (60, 180))
        if lo <= bpm <= hi:
            # Score higher for tracks in the middle of the range
            mid = (lo + hi) / 2
            score = 1.0 - abs(bpm - mid) / (hi - lo + 1)
            return round(score, 4), f"BPM {bpm} fits {pace} range ({lo}–{hi})"
        # Partial credit for just outside the range
        overshoot = max(0, lo - bpm, bpm - hi)
        score = max(0.0, 1.0 - overshoot / 50)
        return round(score, 4), f"BPM {bpm} near {pace} range ({lo}–{hi})"

    @staticmethod
    def _energy_from_pace(pace: str, mood: str) -> float:
        base = {"slow": 0.25, "moderate": 0.55, "fast": 0.85}.get(pace, 0.55)
        boost = {"energetic": 0.10, "tense": 0.05, "calm": -0.15, "melancholic": -0.10}.get(mood, 0.0)
        return max(0.0, min(1.0, base + boost))

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def summarise(output: RecommendationOutput) -> str:
        lines = [f"[Recommender] Top {len(output.recommendations)} recommendation(s):"]
        for i, r in enumerate(output.recommendations, 1):
            lines.append(f"  {i}. {r}")
        return "\n".join(lines)
