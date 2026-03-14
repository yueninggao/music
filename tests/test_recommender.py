"""
tests/test_recommender.py
Tests for the MusicRecommender (DECISION-MAKING & RECOMMENDATION stage).
"""

import pytest
from src.classifier import ClassificationResult, LabelScore
from src.recommender import (
    MusicRecommender,
    MusicRecommendation,
    RecommendationOutput,
    Track,
    DEFAULT_CATALOGUE,
)


def make_cr(
    scene="outdoor",
    activity="relaxing",
    mood="calm",
    pace="slow",
    lighting="bright",
    hash_val="testhash",
) -> ClassificationResult:
    cr = ClassificationResult(source_hash=hash_val)
    cr.scenes = [LabelScore(scene, 0.8)]
    cr.activities = [LabelScore(activity, 0.75)]
    cr.moods = [LabelScore(mood, 0.85)]
    cr.pace = [LabelScore(pace, 0.90)]
    cr.lighting = [LabelScore(lighting, 0.70)]
    return cr


class TestRecommenderInit:
    def test_default_catalogue_loaded(self):
        rec = MusicRecommender()
        assert len(rec.catalogue) > 0

    def test_custom_catalogue(self):
        custom = [
            Track("c001", "Test Track", "Test Artist", "Pop", "Indie Pop",
                  120, 0.7, 0.8, ["happy"], ["outdoor"], ["dancing"])
        ]
        rec = MusicRecommender(catalogue=custom)
        assert len(rec.catalogue) == 1

    def test_default_top_n(self):
        rec = MusicRecommender()
        assert rec.top_n == 5

    def test_custom_top_n(self):
        rec = MusicRecommender(top_n=3)
        assert rec.top_n == 3


class TestRecommendOutput:
    def setup_method(self):
        self.rec = MusicRecommender(top_n=5)

    def test_returns_recommendation_output(self):
        cr = make_cr()
        out = self.rec.recommend(cr)
        assert isinstance(out, RecommendationOutput)

    def test_recommendation_count(self):
        cr = make_cr()
        out = self.rec.recommend(cr)
        assert len(out.recommendations) <= 5

    def test_recommendations_are_sorted_by_score(self):
        cr = make_cr()
        out = self.rec.recommend(cr)
        scores = [r.score for r in out.recommendations]
        assert scores == sorted(scores, reverse=True)

    def test_all_scores_in_range(self):
        cr = make_cr()
        out = self.rec.recommend(cr)
        for r in out.recommendations:
            assert 0.0 <= r.score <= 1.0

    def test_source_hash_preserved(self):
        cr = make_cr(hash_val="myhash")
        out = self.rec.recommend(cr)
        assert out.source_hash == "myhash"

    def test_reasons_populated(self):
        cr = make_cr()
        out = self.rec.recommend(cr)
        # At least the top recommendation should have reasons
        assert len(out.recommendations[0].reasons) > 0

    def test_top_property(self):
        cr = make_cr()
        out = self.rec.recommend(cr)
        assert out.top is out.recommendations[0]

    def test_top_none_if_empty(self):
        out = RecommendationOutput(source_hash="x", classification_summary="")
        assert out.top is None


class TestRecommendationScoring:
    def test_calm_video_prefers_low_energy_tracks(self):
        cr = make_cr(mood="calm", pace="slow", activity="relaxing")
        rec = MusicRecommender(top_n=3)
        out = rec.recommend(cr)
        avg_energy = sum(r.track.energy for r in out.recommendations) / len(out.recommendations)
        assert avg_energy < 0.80, f"Expected lower energy tracks, got avg {avg_energy:.2f}"

    def test_energetic_video_prefers_high_energy_tracks(self):
        cr = make_cr(mood="energetic", pace="fast", activity="dancing",
                     scene="concert", lighting="bright")
        rec = MusicRecommender(top_n=3)
        out = rec.recommend(cr)
        avg_energy = sum(r.track.energy for r in out.recommendations) / len(out.recommendations)
        assert avg_energy > 0.50, f"Expected higher energy tracks, got avg {avg_energy:.2f}"

    def test_tag_match_returns_zero_for_empty(self):
        score, label = MusicRecommender._tag_match([], ["tag1"], [])
        assert score == 0.0
        assert label == ""

    def test_tag_match_full_match(self):
        score, label = MusicRecommender._tag_match(["happy"], ["happy"], [1.0])
        assert score > 0.0
        assert label == "happy"

    def test_bpm_match_within_range(self):
        score, reason = MusicRecommender._bpm_match(100, "moderate")
        assert score > 0.5
        assert "BPM" in reason

    def test_bpm_match_outside_range_partial(self):
        score, _ = MusicRecommender._bpm_match(200, "slow")
        assert score < 1.0  # penalised for being out of range

    def test_energy_from_pace_fast(self):
        e = MusicRecommender._energy_from_pace("fast", "energetic")
        assert e > 0.80

    def test_energy_from_pace_slow(self):
        e = MusicRecommender._energy_from_pace("slow", "calm")
        assert e < 0.40

    def test_top_n_respected(self):
        cr = make_cr()
        rec = MusicRecommender(top_n=2)
        out = rec.recommend(cr)
        assert len(out.recommendations) <= 2


class TestAsDict:
    def test_as_dict_keys(self):
        cr = make_cr()
        rec = MusicRecommender(top_n=3)
        out = rec.recommend(cr)
        d = out.as_dict()
        assert "source_hash" in d
        assert "recommendations" in d
        assert isinstance(d["recommendations"], list)

    def test_recommendation_dict_fields(self):
        cr = make_cr()
        rec = MusicRecommender(top_n=1)
        out = rec.recommend(cr)
        r_dict = out.as_dict()["recommendations"][0]
        for key in ("rank", "title", "artist", "genre", "bpm", "score", "reasons"):
            assert key in r_dict


class TestSummarise:
    def test_summarise_returns_string(self):
        cr = make_cr()
        rec = MusicRecommender(top_n=3)
        out = rec.recommend(cr)
        s = MusicRecommender.summarise(out)
        assert isinstance(s, str)
        assert "Recommender" in s

    def test_music_recommendation_str(self):
        r = MusicRecommendation(
            track=DEFAULT_CATALOGUE[0],
            score=0.85,
            reasons=["mood: energetic"],
        )
        assert str(r)


class TestCatalogueIntegrity:
    def test_all_catalogue_tracks_have_required_fields(self):
        for track in DEFAULT_CATALOGUE:
            assert track.id
            assert track.title
            assert track.artist
            assert track.genre
            assert 0 < track.bpm < 300
            assert 0.0 <= track.energy <= 1.0
            assert 0.0 <= track.valence <= 1.0
