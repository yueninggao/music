"""Music genre classifier based on audio features.

The classifier uses genre-representative feature profiles derived from
well-known acoustic properties of each genre.  Given an input audio file it
extracts features with :mod:`music_classifier.features` and returns the most
likely genre together with a confidence score.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .features import extract_features, features_to_vector

# ---------------------------------------------------------------------------
# Genre definitions and their representative acoustic profiles
# ---------------------------------------------------------------------------

# Each profile lists (name, description, profile_vector_kwargs).
# The profile vectors are deliberately crafted from known acoustic properties:
#   tempo, spectral_centroid_mean, spectral_rolloff_mean, zero_crossing_mean,
#   rms_mean, mfcc_mean[0..2], chroma variance
#
# We generate synthetic training samples around each profile so that a
# StandardScaler + cosine-similarity lookup can work without an actual labelled
# dataset.

GENRES: list[str] = [
    "blues",
    "classical",
    "country",
    "disco",
    "hip-hop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

# Descriptions shown to the user.
GENRE_DESCRIPTIONS: dict[str, str] = {
    "blues": "Blues – soulful, expressive, rooted in African-American traditions",
    "classical": "Classical – orchestral, complex harmonics, wide dynamic range",
    "country": "Country – storytelling lyrics, acoustic guitar, moderate tempo",
    "disco": "Disco – danceable, steady four-on-the-floor beat, lush production",
    "hip-hop": "Hip-Hop – heavy bass, rhythmic speech, sample-based production",
    "jazz": "Jazz – improvisation, complex chords, swing rhythm",
    "metal": "Metal – high energy, distorted guitars, fast tempo",
    "pop": "Pop – catchy melodies, polished production, broad appeal",
    "reggae": "Reggae – off-beat rhythm, bass-heavy, Caribbean roots",
    "rock": "Rock – electric guitars, strong backbeat, powerful vocals",
}

# ---------------------------------------------------------------------------
# Synthetic genre profiles
# (indices match features_to_vector output layout)
# Feature vector layout (58 dimensions total):
#   [0:13]  mfcc_mean
#   [13:26] mfcc_std
#   [26:38] chroma_mean
#   [38:50] chroma_std
#   [50]    spectral_centroid_mean
#   [51]    spectral_centroid_std
#   [52]    spectral_rolloff_mean
#   [53]    spectral_bandwidth_mean
#   [54]    zero_crossing_mean
#   [55]    tempo
#   [56]    rms_mean
#   [57]    rms_std
# ---------------------------------------------------------------------------

_FEATURE_DIM = 58  # total dimensions from features_to_vector

# Compact profile: only the scalar features at [50:58] plus MFCC mean[0]
# are used to define genre "centres".  The rest are filled with zeros.
# Values are approximate absolute magnitudes (not normalised).

_PROFILES: dict[str, dict] = {
    #                   sc_mean  sc_std  sr_mean  sb_mean  zcr    tempo  rms_m  rms_s  mfcc0
    "blues":      dict(sc=1600,  sc_s=300, sr=3200, sb=1500, zcr=0.06, t=85,  rms=0.05, rs=0.02, m0=-200),
    "classical":  dict(sc=2200,  sc_s=600, sr=4500, sb=2000, zcr=0.07, t=70,  rms=0.04, rs=0.03, m0=-150),
    "country":    dict(sc=1900,  sc_s=350, sr=3800, sb=1800, zcr=0.08, t=105, rms=0.07, rs=0.025, m0=-180),
    "disco":      dict(sc=2500,  sc_s=400, sr=5000, sb=2200, zcr=0.10, t=120, rms=0.09, rs=0.03, m0=-160),
    "hip-hop":    dict(sc=1400,  sc_s=250, sr=2800, sb=1400, zcr=0.05, t=90,  rms=0.08, rs=0.04, m0=-220),
    "jazz":       dict(sc=2000,  sc_s=550, sr=4000, sb=1900, zcr=0.09, t=130, rms=0.06, rs=0.03, m0=-170),
    "metal":      dict(sc=3200,  sc_s=500, sr=6500, sb=2800, zcr=0.18, t=160, rms=0.12, rs=0.05, m0=-130),
    "pop":        dict(sc=2300,  sc_s=400, sr=4800, sb=2100, zcr=0.11, t=115, rms=0.08, rs=0.025, m0=-155),
    "reggae":     dict(sc=1700,  sc_s=300, sr=3400, sb=1600, zcr=0.07, t=80,  rms=0.06, rs=0.025, m0=-190),
    "rock":       dict(sc=2700,  sc_s=450, sr=5500, sb=2400, zcr=0.14, t=135, rms=0.10, rs=0.04, m0=-140),
}


def _build_profile_vector(p: dict) -> np.ndarray:
    """Convert a compact profile dict to a full feature vector."""
    v = np.zeros(_FEATURE_DIM, dtype=np.float64)
    v[0] = p["m0"]           # mfcc_mean[0]
    v[50] = p["sc"]          # spectral_centroid_mean
    v[51] = p["sc_s"]        # spectral_centroid_std
    v[52] = p["sr"]          # spectral_rolloff_mean
    v[53] = p["sb"]          # spectral_bandwidth_mean
    v[54] = p["zcr"]         # zero_crossing_mean
    v[55] = p["t"]           # tempo
    v[56] = p["rms"]         # rms_mean
    v[57] = p["rs"]          # rms_std
    return v


def _build_training_data(
    n_samples_per_genre: int = 40, noise_scale: float = 0.12, seed: int = 42
) -> tuple[np.ndarray, list[str]]:
    """Generate synthetic training samples by perturbing each genre profile."""
    rng = np.random.default_rng(seed)
    X: list[np.ndarray] = []
    y: list[str] = []
    for genre in GENRES:
        centre = _build_profile_vector(_PROFILES[genre])
        for _ in range(n_samples_per_genre):
            noise = rng.normal(0, noise_scale * (np.abs(centre) + 1e-6))
            X.append(centre + noise)
            y.append(genre)
    return np.array(X), y


class MusicClassifier:
    """Classifies a music clip into one of ten genres.

    Usage
    -----
    >>> clf = MusicClassifier()
    >>> genre, confidence, scores = clf.classify("path/to/song.mp3")
    >>> print(genre, f"{confidence:.1%}")
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        X, y = _build_training_data()
        self._X_train_scaled = self._scaler.fit_transform(X)
        self._y_train = y
        # Pre-compute per-genre centres in scaled space for fast lookup
        self._centres: dict[str, np.ndarray] = {}
        for genre in GENRES:
            mask = np.array([g == genre for g in y])
            self._centres[genre] = self._X_train_scaled[mask].mean(axis=0)

    def classify(self, audio_path: str, duration: float | None = 30.0) -> tuple[str, float, dict[str, float]]:
        """Classify the genre of an audio file.

        Parameters
        ----------
        audio_path:
            Path to the audio file.
        duration:
            Seconds to analyse (default 30 s, use ``None`` for the full file).

        Returns
        -------
        genre : str
            Predicted genre label.
        confidence : float
            Cosine-similarity score in ``[0, 1]`` for the top genre.
        scores : dict[str, float]
            Similarity score for every genre, sorted highest to lowest.
        """
        raw_features = extract_features(audio_path, duration=duration)
        vec = features_to_vector(raw_features)
        vec_scaled = self._scaler.transform(vec.reshape(1, -1))[0]

        scores: dict[str, float] = {}
        for genre, centre in self._centres.items():
            norm = np.linalg.norm(vec_scaled) * np.linalg.norm(centre)
            if norm < 1e-10:
                scores[genre] = 0.0
            else:
                cosine_sim = float(np.dot(vec_scaled, centre) / norm)
                # Map [-1, 1] → [0, 1]
                scores[genre] = (cosine_sim + 1.0) / 2.0

        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        top_genre = next(iter(sorted_scores))
        confidence = sorted_scores[top_genre]
        return top_genre, confidence, sorted_scores
