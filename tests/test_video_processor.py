"""
tests/test_video_processor.py
Tests for the VideoProcessor (INPUT stage).
"""

import pytest
from src.video_processor import VideoProcessor, FrameBundle, FrameFeatures


class TestVideoProcessorInit:
    def test_default_parameters(self):
        vp = VideoProcessor()
        assert vp.sample_interval_sec == 1.0
        assert vp.max_frames == 300
        assert vp.use_opencv is False

    def test_custom_parameters(self):
        vp = VideoProcessor(sample_interval_sec=2.5, max_frames=50)
        assert vp.sample_interval_sec == 2.5
        assert vp.max_frames == 50

    def test_negative_interval_raises(self):
        with pytest.raises(ValueError):
            VideoProcessor(sample_interval_sec=-1.0)

    def test_zero_interval_raises(self):
        with pytest.raises(ValueError):
            VideoProcessor(sample_interval_sec=0.0)

    def test_zero_max_frames_raises(self):
        with pytest.raises(ValueError):
            VideoProcessor(max_frames=0)


class TestSyntheticProcessing:
    def setup_method(self):
        self.vp = VideoProcessor(sample_interval_sec=1.0, max_frames=100)

    def test_returns_frame_bundle(self):
        bundle = self.vp.process("synthetic:10")
        assert isinstance(bundle, FrameBundle)

    def test_correct_duration(self):
        bundle = self.vp.process("synthetic:30")
        assert bundle.duration_sec == 30.0

    def test_correct_frame_count(self):
        # 10 seconds at 1 s/frame → 10 frames
        bundle = self.vp.process("synthetic:10")
        assert bundle.total_frames_sampled == 10
        assert len(bundle.frames) == 10

    def test_max_frames_cap(self):
        vp = VideoProcessor(sample_interval_sec=1.0, max_frames=5)
        bundle = vp.process("synthetic:100")
        assert bundle.total_frames_sampled <= 5

    def test_frame_features_type(self):
        bundle = self.vp.process("synthetic:5")
        for frame in bundle.frames:
            assert isinstance(frame, FrameFeatures)

    def test_brightness_in_range(self):
        bundle = self.vp.process("synthetic:20")
        for f in bundle.frames:
            assert 0.0 <= f.brightness <= 1.0

    def test_motion_in_range(self):
        bundle = self.vp.process("synthetic:20")
        for f in bundle.frames:
            assert 0.0 <= f.motion_score <= 1.0

    def test_color_histogram_sums_to_one(self):
        bundle = self.vp.process("synthetic:20")
        for f in bundle.frames:
            total = sum(f.color_histogram.values())
            assert abs(total - 1.0) < 1e-3, f"Histogram sum {total} != 1.0"

    def test_aggregates_computed(self):
        bundle = self.vp.process("synthetic:10")
        assert 0.0 <= bundle.avg_brightness <= 1.0
        assert 0.0 <= bundle.avg_motion <= 1.0
        assert bundle.content_hash != ""

    def test_deterministic_output(self):
        b1 = self.vp.process("synthetic:15:myseed")
        b2 = self.vp.process("synthetic:15:myseed")
        assert b1.avg_brightness == b2.avg_brightness
        assert b1.content_hash == b2.content_hash

    def test_different_seeds_differ(self):
        b1 = self.vp.process("synthetic:15:seedA")
        b2 = self.vp.process("synthetic:15:seedB")
        # Very unlikely to be equal with different seeds
        assert b1.avg_brightness != b2.avg_brightness

    def test_fallback_for_unknown_path(self):
        bundle = self.vp.process("/some/nonexistent/video.mp4")
        assert isinstance(bundle, FrameBundle)
        assert bundle.total_frames_sampled > 0

    def test_timestamps_are_sequential(self):
        bundle = self.vp.process("synthetic:10")
        timestamps = [f.timestamp_sec for f in bundle.frames]
        assert timestamps == sorted(timestamps)

    def test_first_frame_zero_motion(self):
        bundle = self.vp.process("synthetic:5")
        assert bundle.frames[0].motion_score == 0.0


class TestFrameBundleAggregates:
    def test_empty_bundle_no_crash(self):
        bundle = FrameBundle(
            source_path="test", total_frames_sampled=0, duration_sec=0.0
        )
        bundle.compute_aggregates()
        assert bundle.avg_brightness == 0.0

    def test_dominant_color_computed(self):
        vp = VideoProcessor()
        bundle = vp.process("synthetic:10")
        assert "R" in bundle.dominant_color
        assert "G" in bundle.dominant_color
        assert "B" in bundle.dominant_color


class TestSummarise:
    def test_summarise_returns_string(self):
        vp = VideoProcessor()
        bundle = vp.process("synthetic:5")
        s = VideoProcessor.summarise(bundle)
        assert isinstance(s, str)
        assert "VideoProcessor" in s
