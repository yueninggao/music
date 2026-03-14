"""
creative_pipeline.py
====================
CREATIVE STEPS AFTER RECOMMENDATION stage of the pipeline.

Given a RecommendationOutput (the ranked music list) this module applies a
set of creative post-processing steps to produce a rich, actionable creative
package:

  1. PlaylistCurator   – selects a coherent playlist ordering and adds
                         transitional metadata (key compatibility, energy arc)
  2. TempoMapper       – maps video scene timestamps to recommended BPM changes
  3. MoodArcBuilder    – builds a mood arc narrative for the full video duration
  4. CreativeNoteWriter – generates a human-readable creative brief
  5. CreativePipeline  – orchestrates all four steps and returns a
                         CreativePackage

The CreativePackage is the final OUTPUT of the entire system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .recommender import RecommendationOutput, MusicRecommendation, Track


# ---------------------------------------------------------------------------
# Supporting data containers
# ---------------------------------------------------------------------------

@dataclass
class PlaylistEntry:
    """One entry in the curated playlist with extra creative metadata."""
    position: int
    track: Track
    transition_from_prev: str   # e.g. "key modulation up a fifth"
    energy_delta: float         # change in energy vs. previous track (–1 to +1)
    cue_note: str               # human note for editor / director


@dataclass
class TempoSegment:
    """A time segment in the video with an assigned BPM target."""
    start_sec: float
    end_sec: float
    bpm_target: int
    label: str          # e.g. "intro", "build", "climax", "outro"
    track_id: str       # which track should play here


@dataclass
class MoodArcPoint:
    """One point on the mood arc timeline."""
    timestamp_sec: float
    mood_label: str
    intensity: float    # 0.0 – 1.0


@dataclass
class CreativePackage:
    """
    The complete creative output package produced by the CreativePipeline.
    This is the final OUTPUT of the full system.
    """
    source_hash: str
    playlist: List[PlaylistEntry] = field(default_factory=list)
    tempo_map: List[TempoSegment] = field(default_factory=list)
    mood_arc: List[MoodArcPoint] = field(default_factory=list)
    creative_brief: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_hash": self.source_hash,
            "playlist": [
                {
                    "position": e.position,
                    "title": e.track.title,
                    "artist": e.track.artist,
                    "genre": e.track.genre,
                    "bpm": e.track.bpm,
                    "energy": e.track.energy,
                    "transition_from_prev": e.transition_from_prev,
                    "energy_delta": round(e.energy_delta, 3),
                    "cue_note": e.cue_note,
                }
                for e in self.playlist
            ],
            "tempo_map": [
                {
                    "start_sec": s.start_sec,
                    "end_sec": s.end_sec,
                    "bpm_target": s.bpm_target,
                    "label": s.label,
                    "track_id": s.track_id,
                }
                for s in self.tempo_map
            ],
            "mood_arc": [
                {
                    "timestamp_sec": p.timestamp_sec,
                    "mood_label": p.mood_label,
                    "intensity": round(p.intensity, 3),
                }
                for p in self.mood_arc
            ],
            "creative_brief": self.creative_brief,
        }

    def __str__(self) -> str:
        lines = [
            "━━━ CREATIVE PACKAGE ━━━",
            f"Source: {self.source_hash}",
            "",
            "▸ PLAYLIST",
        ]
        for e in self.playlist:
            lines.append(
                f"  {e.position}. {e.track.title} – {e.track.artist}"
                f"  [{e.track.bpm} BPM, energy={e.track.energy:.2f}]"
                f"  | {e.cue_note}"
            )
        lines += ["", "▸ TEMPO MAP"]
        for s in self.tempo_map:
            lines.append(
                f"  {s.start_sec:.0f}s–{s.end_sec:.0f}s  {s.label:10s}  {s.bpm_target} BPM  ({s.track_id})"
            )
        lines += ["", "▸ MOOD ARC (keyframes)"]
        for p in self.mood_arc:
            bar = "█" * int(p.intensity * 20)
            lines.append(f"  t={p.timestamp_sec:5.0f}s  {p.mood_label:12s}  {bar}")
        lines += ["", "▸ CREATIVE BRIEF", self.creative_brief]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1 – PlaylistCurator
# ---------------------------------------------------------------------------

_MUSICAL_KEYS = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]

_TRANSITION_PHRASES = [
    "smooth energy fade",
    "beat-matched crossfade",
    "key modulation up a fourth",
    "half-time bridge",
    "energy build-up drop",
    "ambient pad transition",
    "tempo gradual increase",
    "emotional peak handoff",
    "instrumental break",
    "harmonic pivot",
]


class PlaylistCurator:
    """
    Decides the optimal ordering of recommended tracks and adds transition
    metadata between each consecutive pair.

    Strategy
    --------
    • Start with moderate energy to set the scene.
    • Build towards the highest energy track at ~60–70 % of the way through.
    • Cool down towards the end.
    • Assign a musically descriptive transition between each pair.
    """

    def curate(self, recs: List[MusicRecommendation]) -> List[PlaylistEntry]:
        if not recs:
            return []

        tracks = [r.track for r in recs]
        ordered = self._energy_arc_sort(tracks)

        entries: List[PlaylistEntry] = []
        for i, track in enumerate(ordered):
            prev = ordered[i - 1] if i > 0 else None
            delta = (track.energy - prev.energy) if prev else 0.0
            transition = self._choose_transition(prev, track, i) if prev else "opening track"
            cue = self._cue_note(track, i, len(ordered))
            entries.append(
                PlaylistEntry(
                    position=i + 1,
                    track=track,
                    transition_from_prev=transition,
                    energy_delta=round(delta, 3),
                    cue_note=cue,
                )
            )
        return entries

    @staticmethod
    def _energy_arc_sort(tracks: List[Track]) -> List[Track]:
        """Sort tracks so energy rises to ~60 % then falls."""
        if len(tracks) <= 2:
            return sorted(tracks, key=lambda t: t.energy)
        peak_idx = max(1, int(len(tracks) * 0.6))
        sorted_tracks = sorted(tracks, key=lambda t: t.energy)
        # Build an arc: ascending to peak, then descending
        ascending = sorted_tracks[:peak_idx]
        descending = list(reversed(sorted_tracks[peak_idx:]))
        return ascending + descending

    @staticmethod
    def _choose_transition(prev: Track, nxt: Track, idx: int) -> str:
        delta = nxt.energy - prev.energy
        bpm_diff = abs(nxt.bpm - prev.bpm)
        if bpm_diff < 8:
            return "beat-matched crossfade"
        if delta > 0.20:
            return "energy build-up drop"
        if delta < -0.20:
            return "ambient pad transition"
        if bpm_diff < 20:
            return "smooth energy fade"
        return _TRANSITION_PHRASES[idx % len(_TRANSITION_PHRASES)]

    @staticmethod
    def _cue_note(track: Track, idx: int, total: int) -> str:
        if idx == 0:
            return "Scene setter – establish atmosphere"
        if idx == total - 1:
            return "Closing – allow natural fade out"
        if track.energy > 0.80:
            return "Peak moment – sync to visual climax"
        if track.energy < 0.35:
            return "Reflective segment – let visuals breathe"
        return "Mid-section – sustain narrative momentum"


# ---------------------------------------------------------------------------
# Step 2 – TempoMapper
# ---------------------------------------------------------------------------

_ARC_LABELS = ["intro", "build", "climax", "resolution", "outro"]


class TempoMapper:
    """
    Divides the video duration into labelled segments and assigns a BPM target
    and track reference to each segment.
    """

    def map(
        self,
        duration_sec: float,
        playlist: List[PlaylistEntry],
    ) -> List[TempoSegment]:
        if not playlist:
            return []

        n_segments = min(len(_ARC_LABELS), len(playlist) + 1)
        seg_dur = duration_sec / n_segments
        segments: List[TempoSegment] = []

        for i in range(n_segments):
            start = round(i * seg_dur, 1)
            end = round(min((i + 1) * seg_dur, duration_sec), 1)
            label = _ARC_LABELS[i % len(_ARC_LABELS)]

            # Assign track based on position; cycle if needed
            entry = playlist[min(i, len(playlist) - 1)]
            bpm_target = entry.track.bpm

            # Slight BPM modulation per arc stage
            modulation = {
                "intro": -8, "build": 0, "climax": +5,
                "resolution": -10, "outro": -15,
            }.get(label, 0)
            bpm_target = max(50, bpm_target + modulation)

            segments.append(
                TempoSegment(
                    start_sec=start,
                    end_sec=end,
                    bpm_target=bpm_target,
                    label=label,
                    track_id=entry.track.id,
                )
            )
        return segments


# ---------------------------------------------------------------------------
# Step 3 – MoodArcBuilder
# ---------------------------------------------------------------------------

_ARC_MOODS = [
    ("calm",      0.30),
    ("building",  0.55),
    ("energetic", 0.90),
    ("reflective",0.55),
    ("calm",      0.25),
]


class MoodArcBuilder:
    """Builds a mood arc (timeline of mood keyframes) for the video."""

    def build(self, duration_sec: float, top_mood: str) -> List[MoodArcPoint]:
        if duration_sec <= 0:
            return []

        # Override middle keyframe with the video's detected top mood
        arc = list(_ARC_MOODS)
        if len(arc) >= 3:
            arc[2] = (top_mood, arc[2][1])

        points: List[MoodArcPoint] = []
        for i, (mood, intensity) in enumerate(arc):
            t = round((i / max(len(arc) - 1, 1)) * duration_sec, 1)
            points.append(MoodArcPoint(timestamp_sec=t, mood_label=mood, intensity=intensity))

        # Interpolate extra keyframes for richer visualisation
        return self._interpolate(points, duration_sec)

    @staticmethod
    def _interpolate(
        keyframes: List[MoodArcPoint], duration_sec: float, steps: int = 4
    ) -> List[MoodArcPoint]:
        """Insert linearly interpolated intensity points between keyframes."""
        if len(keyframes) < 2:
            return keyframes
        out: List[MoodArcPoint] = []
        for i in range(len(keyframes) - 1):
            p0, p1 = keyframes[i], keyframes[i + 1]
            out.append(p0)
            dt = (p1.timestamp_sec - p0.timestamp_sec) / (steps + 1)
            for s in range(1, steps + 1):
                t = round(p0.timestamp_sec + s * dt, 1)
                intensity = p0.intensity + (p1.intensity - p0.intensity) * (s / (steps + 1))
                # Interpolate mood label based on intensity
                mood = p0.mood_label if s <= steps // 2 else p1.mood_label
                out.append(MoodArcPoint(timestamp_sec=t, mood_label=mood, intensity=round(intensity, 3)))
        out.append(keyframes[-1])
        return out


# ---------------------------------------------------------------------------
# Step 4 – CreativeNoteWriter
# ---------------------------------------------------------------------------

class CreativeNoteWriter:
    """
    Generates a human-readable creative brief for an editor / music supervisor.
    """

    def write(
        self,
        reco_output: RecommendationOutput,
        playlist: List[PlaylistEntry],
        tempo_map: List[TempoSegment],
        mood_arc: List[MoodArcPoint],
    ) -> str:
        top = reco_output.top
        if not top:
            return "No recommendations available to form a creative brief."

        genre = top.track.genre
        mood_peak = max(mood_arc, key=lambda p: p.intensity).mood_label if mood_arc else "energetic"
        bpm_range_lo = min(e.track.bpm for e in playlist) if playlist else top.track.bpm
        bpm_range_hi = max(e.track.bpm for e in playlist) if playlist else top.track.bpm
        n_tracks = len(playlist)
        opening = playlist[0].track.title if playlist else top.track.title
        closing = playlist[-1].track.title if playlist else top.track.title

        classification_clean = reco_output.classification_summary.replace("[Classification] ", "")

        brief = f"""CREATIVE MUSIC BRIEF
=====================
Video classification : {classification_clean}
Primary genre        : {genre}
Emotional peak       : {mood_peak}
BPM range            : {bpm_range_lo}–{bpm_range_hi} BPM
Tracks selected      : {n_tracks}

PLAYLIST NARRATIVE
------------------
The soundtrack opens with "{opening}", establishing the visual tone and
inviting the viewer into the scene. As the action develops, energy builds
progressively through {n_tracks - 2 if n_tracks > 2 else "successive"} transitional tracks,
reaching its emotional climax with the highest-energy selection before
cooling into a reflective resolution. The sequence closes with "{closing}",
allowing space for the final image to resonate.

SYNC NOTES
----------
{self._sync_notes(tempo_map)}

CREATIVE DIRECTION
------------------
{self._creative_direction(mood_peak, genre, top.track)}
"""
        return brief.strip()

    @staticmethod
    def _sync_notes(tempo_map: List[TempoSegment]) -> str:
        if not tempo_map:
            return "No tempo map available."
        lines = []
        for seg in tempo_map:
            lines.append(
                f"  • {seg.label.capitalize():12s}  {seg.start_sec:.0f}s → {seg.end_sec:.0f}s"
                f"  @ {seg.bpm_target} BPM  [{seg.track_id}]"
            )
        return "\n".join(lines)

    @staticmethod
    def _creative_direction(mood: str, genre: str, top_track: Track) -> str:
        directions = {
            "energetic": (
                "Use quick cuts and dynamic camera moves in sync with the beat. "
                "Align major visual transitions to bar boundaries."
            ),
            "calm": (
                "Allow longer takes and slow dissolves. Let the music breathe "
                "between scenes rather than forcing hard cuts."
            ),
            "melancholic": (
                "Favour desaturated colour grading and wide, lonely compositions. "
                "Let the music's natural reverb tail carry between shots."
            ),
            "romantic": (
                "Soft focus, warm colour palette, and gentle cross-dissolves. "
                "Music swells should coincide with intimate close-up moments."
            ),
            "happy": (
                "Bright, saturated visuals with upbeat tempo cuts. Match the "
                "percussive hits to action beats for a celebratory feel."
            ),
            "tense": (
                "High contrast grading, tight framing, and jarring cuts on off-beats. "
                "Use silence or drop-outs before explosive moments."
            ),
        }
        base = directions.get(
            mood,
            "Match edit rhythm to the natural groove of the selected music. "
            "Allow the musical structure to guide pacing decisions.",
        )
        return (
            f"{base}\n\n"
            f"  Anchor track: \"{top_track.title}\" by {top_track.artist}\n"
            f"  Genre: {genre}  |  BPM: {top_track.bpm}  |  Energy: {top_track.energy:.2f}"
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class CreativePipeline:
    """
    Orchestrates the four creative post-processing steps and produces a
    ``CreativePackage`` – the final OUTPUT of the full music-video system.

    Parameters
    ----------
    top_n_playlist : int
        How many recommended tracks to include in the curated playlist.
    duration_sec : float
        Duration of the source video (passed through from FrameBundle).
    """

    def __init__(self, top_n_playlist: int = 5, duration_sec: float = 60.0) -> None:
        self.top_n_playlist = top_n_playlist
        self.duration_sec = duration_sec
        self._curator = PlaylistCurator()
        self._tempo_mapper = TempoMapper()
        self._mood_arc_builder = MoodArcBuilder()
        self._note_writer = CreativeNoteWriter()

    def run(self, reco_output: RecommendationOutput, top_mood: str = "energetic") -> CreativePackage:
        """
        Execute all creative post-processing steps.

        Parameters
        ----------
        reco_output : RecommendationOutput
            Output from MusicRecommender.
        top_mood : str
            Top detected mood from ClassificationResult (used in mood arc).

        Returns
        -------
        CreativePackage
        """
        recs = reco_output.recommendations[: self.top_n_playlist]

        # Step 1 – curate playlist
        playlist = self._curator.curate(recs)

        # Step 2 – build tempo map
        tempo_map = self._tempo_mapper.map(self.duration_sec, playlist)

        # Step 3 – build mood arc
        mood_arc = self._mood_arc_builder.build(self.duration_sec, top_mood)

        # Step 4 – write creative brief
        brief = self._note_writer.write(reco_output, playlist, tempo_map, mood_arc)

        return CreativePackage(
            source_hash=reco_output.source_hash,
            playlist=playlist,
            tempo_map=tempo_map,
            mood_arc=mood_arc,
            creative_brief=brief,
        )
