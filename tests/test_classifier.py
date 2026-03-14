"""
tests/test_classifier.py
Tests for the VideoClassifier (RECOGNITION & CLASSIFICATION stage).
"""

import pytest
from src.video_processor import VideoProcessor, FrameBundle
from src.classifier import VideoClassifier, ClassificationResult, LabelScore


def make_bundle(brightness=0.5, motion=0.2, edge=0.4, duration=30.0) -> FrameBundle:
    """Create a FrameBundle with controlled aggregate values for testing."""
    vp = VideoProcessor(sample_interval_sec=1.0, max_frames=1)
    bundle = vp.process(f"synthetic:{duration}")
    # Override aggregates directly for deterministic tests
    bundle.avg_brightness = brightness
    bundle.avg_motion = motion
    bundle.avg_edge_density = edge
    bundle.dominant_color = {"R": 0.33, "G": 0.34, "B": 0.33}
    bundle.content_hash = "testhash001"
    return bundle


class TestClassifierInit:
    def test_default_init(self):
        vc = VideoClassifier()
        assert vc.model_fn is None
        assert vc.min_confidence > 0.0

    def test_custom_min_confidence(self):
        vc = VideoClassifier(min_confidence=0.20)
        assert vc.min_confidence == 0.20


class TestClassificationResult:
    def setup_method(self):
        self.vc = VideoClassifier()

    def test_returns_classification_result(self):
        bundle = make_bundle()
        result = self.vc.classify(bundle)
        assert isinstance(result, ClassificationResult)

    def test_source_hash_preserved(self):
        bundle = make_bundle()
        result = self.vc.classify(bundle)
        assert result.source_hash == bundle.content_hash

    def test_has_scenes(self):
        bundle = make_bundle()
        result = self.vc.classify(bundle)
        assert len(result.scenes) > 0

    def test_has_moods(self):
        bundle = make_bundle()
        result = self.vc.classify(bundle)
        assert len(result.moods) > 0

    def test_has_activities(self):
        bundle = make_bundle()
        result = self.vc.classify(bundle)
        assert len(result.activities) > 0

    def test_has_pace(self):
        bundle = make_bundle()
        result = self.vc.classify(bundle)
        assert len(result.pace) > 0

    def test_has_lighting(self):
        bundle = make_bundle()
        result = self.vc.classify(bundle)
        assert len(result.lighting) > 0

    def test_confidence_in_range(self):
        bundle = make_bundle()
        result = self.vc.classify(bundle)
        for ls in result.scenes + result.moods + result.activities:
            assert 0.0 <= ls.confidence <= 1.0

    def test_top_properties_are_strings(self):
        bundle = make_bundle()
        result = self.vc.classify(bundle)
        assert isinstance(result.top_scene, str)
        assert isinstance(result.top_mood, str)
        assert isinstance(result.top_activity, str)
        assert isinstance(result.top_pace, str)
        assert isinstance(result.top_lighting, str)

    def test_as_dict_structure(self):
        bundle = make_bundle()
        result = self.vc.classify(bundle)
        d = result.as_dict()
        assert "scene" in d
        assert "mood" in d
        assert "activity" in d
        assert "pace" in d

    def test_str_representation(self):
        bundle = make_bundle()
        result = self.vc.classify(bundle)
        s = str(result)
        assert "Classification" in s


class TestClassificationByMetrics:
    def setup_method(self):
        self.vc = VideoClassifier()

    def test_high_motion_energetic_mood(self):
        bundle = make_bundle(brightness=0.7, motion=0.45, edge=0.6)
        result = self.vc.classify(bundle)
        # Should have energetic or tense mood (high motion + high brightness)
        mood_labels = [ls.label for ls in result.moods]
        assert any(m in mood_labels for m in ("energetic", "happy", "tense"))

    def test_low_brightness_dark_lighting(self):
        bundle = make_bundle(brightness=0.20, motion=0.1, edge=0.3)
        result = self.vc.classify(bundle)
        lighting_labels = [ls.label for ls in result.lighting]
        assert "dark" in lighting_labels

    def test_high_brightness_bright_lighting(self):
        bundle = make_bundle(brightness=0.80, motion=0.1, edge=0.3)
        result = self.vc.classify(bundle)
        lighting_labels = [ls.label for ls in result.lighting]
        assert "bright" in lighting_labels

    def test_fast_pace_for_high_motion(self):
        bundle = make_bundle(brightness=0.5, motion=0.45, edge=0.5)
        result = self.vc.classify(bundle)
        assert result.top_pace == "fast"

    def test_slow_pace_for_low_motion(self):
        bundle = make_bundle(brightness=0.5, motion=0.05, edge=0.3)
        result = self.vc.classify(bundle)
        assert result.top_pace == "slow"

    def test_moderate_pace(self):
        bundle = make_bundle(brightness=0.5, motion=0.18, edge=0.4)
        result = self.vc.classify(bundle)
        assert result.top_pace == "moderate"

    def test_calm_mood_for_low_motion_good_brightness(self):
        bundle = make_bundle(brightness=0.60, motion=0.05, edge=0.3)
        result = self.vc.classify(bundle)
        mood_labels = [ls.label for ls in result.moods]
        assert "calm" in mood_labels

    def test_melancholic_mood_for_dark_scene(self):
        bundle = make_bundle(brightness=0.20, motion=0.1, edge=0.3)
        result = self.vc.classify(bundle)
        mood_labels = [ls.label for ls in result.moods]
        assert "melancholic" in mood_labels


class TestModelFusion:
    def test_model_fn_boosts_label(self):
        def fake_model(bundle):
            return {"scene": "concert", "mood": "energetic"}

        vc = VideoClassifier(model_fn=fake_model)
        bundle = make_bundle(motion=0.35, brightness=0.6, edge=0.6)
        result = vc.classify(bundle)
        # The model should have boosted "concert" and "energetic"
        scene_labels = [ls.label for ls in result.scenes]
        assert "concert" in scene_labels

    def test_failing_model_falls_back(self):
        def bad_model(bundle):
            raise RuntimeError("model unavailable")

        vc = VideoClassifier(model_fn=bad_model)
        bundle = make_bundle()
        # Should not raise; rule-based result returned
        result = vc.classify(bundle)
        assert isinstance(result, ClassificationResult)

    def test_model_adds_new_label(self):
        def fake_model(bundle):
            return {"scene": "fantasy_realm"}  # label not in rules

        vc = VideoClassifier(model_fn=fake_model)
        bundle = make_bundle()
        result = vc.classify(bundle)
        scene_labels = [ls.label for ls in result.scenes]
        assert "fantasy_realm" in scene_labels


class TestLabelScore:
    def test_confidence_clamped(self):
        ls = LabelScore(label="test", confidence=1.5)
        assert ls.confidence == 1.0

    def test_confidence_clamped_low(self):
        ls = LabelScore(label="test", confidence=-0.3)
        assert ls.confidence == 0.0


class TestSortingAndDeduplication:
    def test_scenes_sorted_by_confidence(self):
        vc = VideoClassifier()
        bundle = make_bundle()
        result = vc.classify(bundle)
        confs = [ls.confidence for ls in result.scenes]
        assert confs == sorted(confs, reverse=True)

    def test_moods_sorted_by_confidence(self):
        vc = VideoClassifier()
        bundle = make_bundle()
        result = vc.classify(bundle)
        confs = [ls.confidence for ls in result.moods]
        assert confs == sorted(confs, reverse=True)
