"""Tests for the MusicClassifier."""

import numpy as np
import pytest
import soundfile as sf

from music_classifier.classifier import GENRES, MusicClassifier


@pytest.fixture()
def short_noise_file(tmp_path):
    """Write a short random-noise WAV file and return its path."""
    rng = np.random.default_rng(0)
    sr = 22050
    audio = rng.uniform(-0.1, 0.1, size=sr * 3).astype(np.float32)
    path = tmp_path / "noise.wav"
    sf.write(str(path), audio, sr)
    return str(path)


def test_classify_returns_known_genre(short_noise_file):
    """The top predicted genre should be one of the known genres."""
    clf = MusicClassifier()
    genre, _, _ = clf.classify(short_noise_file)
    assert genre in GENRES


def test_classify_confidence_in_range(short_noise_file):
    """Confidence score should lie in [0, 1]."""
    clf = MusicClassifier()
    _, confidence, _ = clf.classify(short_noise_file)
    assert 0.0 <= confidence <= 1.0


def test_classify_scores_all_genres(short_noise_file):
    """Scores dict should contain an entry for every supported genre."""
    clf = MusicClassifier()
    _, _, scores = clf.classify(short_noise_file)
    assert set(scores.keys()) == set(GENRES)


def test_classify_scores_sorted_descending(short_noise_file):
    """Scores dict should be sorted from highest to lowest."""
    clf = MusicClassifier()
    _, _, scores = clf.classify(short_noise_file)
    values = list(scores.values())
    assert values == sorted(values, reverse=True)


def test_classify_top_genre_has_max_score(short_noise_file):
    """The returned genre should correspond to the highest score."""
    clf = MusicClassifier()
    genre, confidence, scores = clf.classify(short_noise_file)
    assert scores[genre] == pytest.approx(confidence)
    assert genre == max(scores, key=scores.__getitem__)
