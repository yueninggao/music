"""
tests/test_pipeline.py
End-to-end integration tests for MusicVideoPipeline.
"""

import json
import pytest
from src.pipeline import MusicVideoPipeline, PipelineResult
from src.video_processor import FrameBundle
from src.classifier import ClassificationResult
from src.recommender import RecommendationOutput, Track
from src.creative_pipeline import CreativePackage


class TestPipelineInit:
    def test_default_init(self):
        p = MusicVideoPipeline()
        assert p._top_n == 5

    def test_custom_top_n(self):
        p = MusicVideoPipeline(top_n_recommendations=3)
        assert p._top_n == 3

    def test_quick_run_classmethod(self):
        result = MusicVideoPipeline.quick_run("synthetic:10", top_n=3)
        assert isinstance(result, PipelineResult)


class TestEndToEndSynthetic:
    def setup_method(self):
        self.pipeline = MusicVideoPipeline(top_n_recommendations=5)

    def test_run_returns_pipeline_result(self):
        result = self.pipeline.run("synthetic:30")
        assert isinstance(result, PipelineResult)

    def test_all_stages_populated(self):
        result = self.pipeline.run("synthetic:30")
        assert result.bundle is not None
        assert result.classification is not None
        assert result.recommendations is not None
        assert result.creative is not None

    def test_bundle_type(self):
        result = self.pipeline.run("synthetic:10")
        assert isinstance(result.bundle, FrameBundle)

    def test_classification_type(self):
        result = self.pipeline.run("synthetic:10")
        assert isinstance(result.classification, ClassificationResult)

    def test_recommendations_type(self):
        result = self.pipeline.run("synthetic:10")
        assert isinstance(result.recommendations, RecommendationOutput)

    def test_creative_type(self):
        result = self.pipeline.run("synthetic:10")
        assert isinstance(result.creative, CreativePackage)

    def test_elapsed_sec_positive(self):
        result = self.pipeline.run("synthetic:10")
        assert result.elapsed_sec > 0.0

    def test_recommendations_count(self):
        pipeline = MusicVideoPipeline(top_n_recommendations=3)
        result = pipeline.run("synthetic:15")
        assert len(result.recommendations.recommendations) <= 3

    def test_source_preserved(self):
        result = self.pipeline.run("synthetic:10")
        assert result.source == "synthetic:10"

    def test_deterministic_for_same_seed(self):
        r1 = self.pipeline.run("synthetic:20:sameseed")
        r2 = self.pipeline.run("synthetic:20:sameseed")
        assert r1.bundle.avg_brightness == r2.bundle.avg_brightness
        assert r1.classification.top_mood == r2.classification.top_mood

    def test_different_sources_can_differ(self):
        r1 = self.pipeline.run("synthetic:20:seedX")
        r2 = self.pipeline.run("synthetic:20:seedY")
        # They share the same pipeline but have different input data
        assert r1.bundle.content_hash != r2.bundle.content_hash

    def test_as_dict_structure(self):
        result = self.pipeline.run("synthetic:10")
        d = result.as_dict()
        assert "source" in d
        assert "bundle" in d
        assert "classification" in d
        assert "recommendations" in d
        assert "creative" in d
        assert "elapsed_sec" in d

    def test_as_dict_serialisable_to_json(self):
        result = self.pipeline.run("synthetic:10")
        d = result.as_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 100

    def test_print_report_no_crash(self, capsys):
        result = self.pipeline.run("synthetic:10")
        result.print_report()
        captured = capsys.readouterr()
        assert "Pipeline Report" in captured.out
        assert "CLASSIFICATION" in captured.out
        assert "RECOMMENDATION" in captured.out

    def test_various_durations(self):
        for duration in [5, 15, 60, 120]:
            result = self.pipeline.run(f"synthetic:{duration}")
            assert result.bundle.duration_sec == duration

    def test_creative_brief_non_empty(self):
        result = self.pipeline.run("synthetic:30")
        assert len(result.creative.creative_brief) > 50

    def test_playlist_length(self):
        pipeline = MusicVideoPipeline(top_n_recommendations=4)
        result = pipeline.run("synthetic:30")
        assert len(result.creative.playlist) <= 4

    def test_tempo_map_covers_duration(self):
        result = self.pipeline.run("synthetic:30")
        if result.creative.tempo_map:
            assert result.creative.tempo_map[-1].end_sec == pytest.approx(30.0, abs=1.0)


class TestModelInjection:
    def test_custom_model_fn_accepted(self):
        called = []

        def my_model(bundle):
            called.append(True)
            return {"scene": "concert", "mood": "energetic"}

        pipeline = MusicVideoPipeline(model_fn=my_model)
        result = pipeline.run("synthetic:10")
        assert len(called) > 0
        # Concert should appear in scenes (possibly boosted)
        scene_labels = [ls.label for ls in result.classification.scenes]
        assert "concert" in scene_labels


class TestCustomCatalogue:
    def test_custom_catalogue_used(self):
        custom = [
            Track("x001", "My Custom Track", "Test Band", "Post-Rock", "Post-Rock",
                  110, 0.65, 0.70, ["calm", "mysterious"], ["outdoor"], ["relaxing"])
        ]
        pipeline = MusicVideoPipeline(custom_catalogue=custom, top_n_recommendations=1)
        result = pipeline.run("synthetic:10")
        recs = result.recommendations.recommendations
        assert len(recs) == 1
        assert recs[0].track.title == "My Custom Track"
