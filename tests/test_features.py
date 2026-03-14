"""Tests for audio feature extraction."""

import numpy as np
import pytest
import soundfile as sf

from music_classifier.features import extract_features, features_to_vector


@pytest.fixture()
def sine_wave_file(tmp_path):
    """Write a short synthetic sine-wave WAV file and return its path."""
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    path = tmp_path / "sine.wav"
    sf.write(str(path), audio, sr)
    return str(path)


def test_extract_features_keys(sine_wave_file):
    """extract_features should return all expected keys."""
    features = extract_features(sine_wave_file)
    expected_keys = {
        "mfcc_mean", "mfcc_std",
        "chroma_mean", "chroma_std",
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_rolloff_mean", "spectral_bandwidth_mean",
        "zero_crossing_mean", "tempo",
        "rms_mean", "rms_std",
    }
    assert expected_keys.issubset(features.keys())


def test_extract_features_shapes(sine_wave_file):
    """MFCC and chroma arrays should have the expected lengths."""
    features = extract_features(sine_wave_file)
    assert features["mfcc_mean"].shape == (13,)
    assert features["mfcc_std"].shape == (13,)
    assert features["chroma_mean"].shape == (12,)
    assert features["chroma_std"].shape == (12,)


def test_extract_features_tempo_nonnegative(sine_wave_file):
    """Tempo should be a non-negative number (0 for signals with no detectable beat)."""
    features = extract_features(sine_wave_file)
    assert features["tempo"] >= 0


def test_features_to_vector_shape(sine_wave_file):
    """features_to_vector should produce a flat 1-D array of length 58."""
    features = extract_features(sine_wave_file)
    vec = features_to_vector(features)
    assert vec.ndim == 1
    assert vec.shape[0] == 58


def test_features_to_vector_dtype(sine_wave_file):
    """Output vector should be float64."""
    features = extract_features(sine_wave_file)
    vec = features_to_vector(features)
    assert vec.dtype == np.float64


def test_extract_features_file_not_found():
    """extract_features should raise when the path does not exist."""
    with pytest.raises(Exception):
        extract_features("/nonexistent/path/to/audio.wav")
