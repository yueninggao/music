"""Tests for the MusicRecommender."""

import pytest

from music_classifier.classifier import GENRES
from music_classifier.recommender import MusicRecommender


@pytest.fixture()
def recommender():
    return MusicRecommender()


def test_recommend_returns_list(recommender):
    """recommend() should return a list."""
    result = recommender.recommend("jazz")
    assert isinstance(result, list)


def test_recommend_default_top_n(recommender):
    """Default top_n=3 should return exactly 3 items."""
    result = recommender.recommend("jazz")
    assert len(result) == 3


def test_recommend_custom_top_n(recommender):
    """Custom top_n should be respected."""
    result = recommender.recommend("rock", top_n=5)
    assert len(result) == 5


@pytest.mark.parametrize("genre", GENRES)
def test_recommend_all_genres(recommender, genre):
    """recommend() should work for every supported genre."""
    result = recommender.recommend(genre, top_n=3)
    assert len(result) == 3


def test_recommend_result_structure(recommender):
    """Each recommendation should have genre, artists, and reason keys."""
    result = recommender.recommend("blues", top_n=1)
    rec = result[0]
    assert "genre" in rec
    assert "artists" in rec
    assert "reason" in rec


def test_recommend_genres_are_valid(recommender):
    """Recommended genres should all be in the supported list."""
    for genre in GENRES:
        for rec in recommender.recommend(genre, top_n=3):
            assert rec["genre"] in GENRES, (
                f"Recommended genre '{rec['genre']}' for '{genre}' is not valid"
            )


def test_recommend_does_not_include_self(recommender):
    """A genre should not recommend itself."""
    for genre in GENRES:
        for rec in recommender.recommend(genre):
            assert rec["genre"] != genre


def test_recommend_artists_nonempty(recommender):
    """Every recommendation should include at least one artist."""
    for rec in recommender.recommend("pop"):
        assert len(rec["artists"]) > 0


def test_recommend_unknown_genre_raises(recommender):
    """Passing an unknown genre should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown genre"):
        recommender.recommend("dubstep")
