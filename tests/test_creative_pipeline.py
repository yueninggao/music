"""
tests/test_creative_pipeline.py
Tests for the CreativePipeline and its sub-components (CREATIVE OUTPUT stage).
"""

import pytest
from src.recommender import MusicRecommender, RecommendationOutput, MusicRecommendation, Track
from src.classifier import ClassificationResult, LabelScore
from src.creative_pipeline import (
    CreativePipeline,
    CreativePackage,
    PlaylistCurator,
    TempoMapper,
    MoodArcBuilder,
    CreativeNoteWriter,
    PlaylistEntry,
    TempoSegment,
    MoodArcPoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reco_output(n_recs: int = 5) -> RecommendationOutput:
    """Build a RecommendationOutput with n_recs synthetic recommendations."""
    cr = ClassificationResult(source_hash="testhash")
    cr.scenes = [LabelScore("outdoor", 0.8)]
    cr.activities = [LabelScore("dancing", 0.7)]
    cr.moods = [LabelScore("energetic", 0.9)]
    cr.pace = [LabelScore("fast", 0.85)]
    cr.lighting = [LabelScore("bright", 0.7)]

    rec = MusicRecommender(top_n=n_recs)
    return rec.recommend(cr)


# ---------------------------------------------------------------------------
# PlaylistCurator tests
# ---------------------------------------------------------------------------

class TestPlaylistCurator:
    def setup_method(self):
        self.curator = PlaylistCurator()

    def test_returns_list(self):
        out = make_reco_output(3)
        entries = self.curator.curate(out.recommendations)
        assert isinstance(entries, list)

    def test_correct_length(self):
        out = make_reco_output(4)
        entries = self.curator.curate(out.recommendations)
        assert len(entries) == 4

    def test_positions_are_sequential(self):
        out = make_reco_output(3)
        entries = self.curator.curate(out.recommendations)
        positions = [e.position for e in entries]
        assert positions == list(range(1, len(entries) + 1))

    def test_opening_track_note(self):
        out = make_reco_output(3)
        entries = self.curator.curate(out.recommendations)
        assert "opener" in entries[0].cue_note.lower() or "scene" in entries[0].cue_note.lower()

    def test_closing_track_note(self):
        out = make_reco_output(3)
        entries = self.curator.curate(out.recommendations)
        assert "clos" in entries[-1].cue_note.lower() or "fade" in entries[-1].cue_note.lower()

    def test_transition_strings_non_empty(self):
        out = make_reco_output(4)
        entries = self.curator.curate(out.recommendations)
        for e in entries:
            assert isinstance(e.transition_from_prev, str)
            assert len(e.transition_from_prev) > 0

    def test_energy_delta_type(self):
        out = make_reco_output(4)
        entries = self.curator.curate(out.recommendations)
        for e in entries:
            assert isinstance(e.energy_delta, float)

    def test_empty_input(self):
        entries = self.curator.curate([])
        assert entries == []

    def test_single_track(self):
        out = make_reco_output(1)
        entries = self.curator.curate(out.recommendations)
        assert len(entries) == 1
        assert entries[0].transition_from_prev == "opening track"


# ---------------------------------------------------------------------------
# TempoMapper tests
# ---------------------------------------------------------------------------

class TestTempoMapper:
    def setup_method(self):
        self.mapper = TempoMapper()

    def _get_playlist(self, n: int = 4) -> list:
        out = make_reco_output(n)
        return PlaylistCurator().curate(out.recommendations)

    def test_returns_list(self):
        playlist = self._get_playlist()
        segments = self.mapper.map(60.0, playlist)
        assert isinstance(segments, list)

    def test_segments_non_empty(self):
        playlist = self._get_playlist()
        segments = self.mapper.map(60.0, playlist)
        assert len(segments) > 0

    def test_segments_cover_full_duration(self):
        playlist = self._get_playlist()
        duration = 90.0
        segments = self.mapper.map(duration, playlist)
        assert segments[-1].end_sec == pytest.approx(duration, abs=0.5)

    def test_segments_are_contiguous(self):
        playlist = self._get_playlist()
        segments = self.mapper.map(60.0, playlist)
        for i in range(1, len(segments)):
            assert segments[i].start_sec == pytest.approx(segments[i - 1].end_sec, abs=0.1)

    def test_bpm_targets_positive(self):
        playlist = self._get_playlist()
        segments = self.mapper.map(60.0, playlist)
        for s in segments:
            assert s.bpm_target > 0

    def test_segment_labels(self):
        playlist = self._get_playlist()
        segments = self.mapper.map(60.0, playlist)
        valid_labels = {"intro", "build", "climax", "resolution", "outro"}
        for s in segments:
            assert s.label in valid_labels

    def test_empty_playlist(self):
        segments = self.mapper.map(60.0, [])
        assert segments == []

    def test_zero_duration(self):
        playlist = self._get_playlist()
        segments = self.mapper.map(0.0, playlist)
        # Segments exist but have start == end == 0
        for s in segments:
            assert s.start_sec == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MoodArcBuilder tests
# ---------------------------------------------------------------------------

class TestMoodArcBuilder:
    def setup_method(self):
        self.builder = MoodArcBuilder()

    def test_returns_list(self):
        arc = self.builder.build(60.0, "energetic")
        assert isinstance(arc, list)

    def test_non_empty(self):
        arc = self.builder.build(60.0, "happy")
        assert len(arc) > 0

    def test_timestamps_in_range(self):
        duration = 60.0
        arc = self.builder.build(duration, "calm")
        for p in arc:
            assert 0.0 <= p.timestamp_sec <= duration

    def test_intensities_in_range(self):
        arc = self.builder.build(60.0, "melancholic")
        for p in arc:
            assert 0.0 <= p.intensity <= 1.0

    def test_mood_labels_are_strings(self):
        arc = self.builder.build(60.0, "tense")
        for p in arc:
            assert isinstance(p.mood_label, str)

    def test_top_mood_appears_in_arc(self):
        arc = self.builder.build(60.0, "romantic")
        labels = {p.mood_label for p in arc}
        assert "romantic" in labels

    def test_zero_duration(self):
        arc = self.builder.build(0.0, "calm")
        assert arc == []


# ---------------------------------------------------------------------------
# CreativeNoteWriter tests
# ---------------------------------------------------------------------------

class TestCreativeNoteWriter:
    def setup_method(self):
        self.writer = CreativeNoteWriter()

    def test_returns_string(self):
        out = make_reco_output()
        playlist = PlaylistCurator().curate(out.recommendations)
        mapper = TempoMapper()
        tempo_map = mapper.map(60.0, playlist)
        mood_arc = MoodArcBuilder().build(60.0, "energetic")
        brief = self.writer.write(out, playlist, tempo_map, mood_arc)
        assert isinstance(brief, str)

    def test_brief_non_empty(self):
        out = make_reco_output()
        playlist = PlaylistCurator().curate(out.recommendations)
        mapper = TempoMapper()
        tempo_map = mapper.map(60.0, playlist)
        mood_arc = MoodArcBuilder().build(60.0, "happy")
        brief = self.writer.write(out, playlist, tempo_map, mood_arc)
        assert len(brief) > 50

    def test_brief_contains_genre(self):
        out = make_reco_output()
        playlist = PlaylistCurator().curate(out.recommendations)
        tempo_map = TempoMapper().map(60.0, playlist)
        mood_arc = MoodArcBuilder().build(60.0, "energetic")
        brief = self.writer.write(out, playlist, tempo_map, mood_arc)
        # Brief should mention the genre
        assert any(g.lower() in brief.lower() for g in ("electronic", "pop", "rock", "indie", "ambient", "r&b", "jazz", "hip-hop", "world", "classical", "folk"))

    def test_no_recs_returns_message(self):
        empty_out = RecommendationOutput(source_hash="x", classification_summary="")
        brief = self.writer.write(empty_out, [], [], [])
        assert "No recommendations" in brief


# ---------------------------------------------------------------------------
# CreativePipeline integration tests
# ---------------------------------------------------------------------------

class TestCreativePipeline:
    def setup_method(self):
        self.pipeline = CreativePipeline(top_n_playlist=5, duration_sec=60.0)

    def test_returns_creative_package(self):
        out = make_reco_output()
        pkg = self.pipeline.run(out, top_mood="energetic")
        assert isinstance(pkg, CreativePackage)

    def test_playlist_populated(self):
        out = make_reco_output()
        pkg = self.pipeline.run(out)
        assert len(pkg.playlist) > 0

    def test_tempo_map_populated(self):
        out = make_reco_output()
        pkg = self.pipeline.run(out)
        assert len(pkg.tempo_map) > 0

    def test_mood_arc_populated(self):
        out = make_reco_output()
        pkg = self.pipeline.run(out)
        assert len(pkg.mood_arc) > 0

    def test_creative_brief_populated(self):
        out = make_reco_output()
        pkg = self.pipeline.run(out)
        assert len(pkg.creative_brief) > 20

    def test_source_hash_in_package(self):
        out = make_reco_output()
        pkg = self.pipeline.run(out)
        assert pkg.source_hash == out.source_hash

    def test_as_dict_structure(self):
        out = make_reco_output()
        pkg = self.pipeline.run(out)
        d = pkg.as_dict()
        assert "playlist" in d
        assert "tempo_map" in d
        assert "mood_arc" in d
        assert "creative_brief" in d

    def test_str_output(self):
        out = make_reco_output()
        pkg = self.pipeline.run(out)
        s = str(pkg)
        assert "CREATIVE PACKAGE" in s
        assert "PLAYLIST" in s
        assert "TEMPO MAP" in s
        assert "MOOD ARC" in s

    def test_empty_recommendations(self):
        empty_out = RecommendationOutput(
            source_hash="empty", classification_summary=""
        )
        pkg = self.pipeline.run(empty_out)
        assert isinstance(pkg, CreativePackage)
        assert pkg.playlist == []

    def test_different_moods_produce_different_briefs(self):
        out = make_reco_output()
        pkg1 = self.pipeline.run(out, top_mood="calm")
        pkg2 = self.pipeline.run(out, top_mood="tense")
        assert pkg1.creative_brief != pkg2.creative_brief
