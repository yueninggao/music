"""Audio feature extraction utilities for music genre classification."""

import numpy as np
import librosa


def extract_features(audio_path: str, duration: float | None = 30.0) -> dict:
    """Extract audio features from a music file.

    Parameters
    ----------
    audio_path:
        Path to the audio file (wav, mp3, flac, ogg, etc.).
    duration:
        Maximum number of seconds to load.  ``None`` loads the full file.

    Returns
    -------
    dict
        A dictionary containing the extracted feature vectors.

    Raises
    ------
    FileNotFoundError
        If *audio_path* does not point to an existing file.
    librosa.util.exceptions.ParameterError
        If the file cannot be decoded as audio.
    """
    y, sr = librosa.load(audio_path, duration=duration, mono=True)

    features: dict = {}

    # --- MFCCs (timbral texture) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features["mfcc_mean"] = np.mean(mfccs, axis=1)
    features["mfcc_std"] = np.std(mfccs, axis=1)

    # --- Chroma (harmonic / pitch class) ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = np.mean(chroma, axis=1)
    features["chroma_std"] = np.std(chroma, axis=1)

    # --- Spectral features ---
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(spec_centroid))
    features["spectral_centroid_std"] = float(np.std(spec_centroid))

    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_rolloff_mean"] = float(np.mean(spec_rolloff))

    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = float(np.mean(spec_bandwidth))

    zero_crossing = librosa.feature.zero_crossing_rate(y)
    features["zero_crossing_mean"] = float(np.mean(zero_crossing))

    # --- Rhythm ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(np.atleast_1d(tempo)[0])

    # --- RMS energy ---
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    return features


def features_to_vector(features: dict) -> np.ndarray:
    """Flatten a feature dictionary into a single 1-D NumPy array.

    Parameters
    ----------
    features:
        Dictionary returned by :func:`extract_features`.

    Returns
    -------
    np.ndarray
        Shape ``(N,)`` float64 array ready for classifier input.
    """
    parts = [
        features["mfcc_mean"],
        features["mfcc_std"],
        features["chroma_mean"],
        features["chroma_std"],
        [
            features["spectral_centroid_mean"],
            features["spectral_centroid_std"],
            features["spectral_rolloff_mean"],
            features["spectral_bandwidth_mean"],
            features["zero_crossing_mean"],
            features["tempo"],
            features["rms_mean"],
            features["rms_std"],
        ],
    ]
    return np.concatenate([np.atleast_1d(p) for p in parts]).astype(np.float64)
