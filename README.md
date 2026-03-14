# Music Genre Classifier 🎵

Analyse a clip of music, classify its genre (jazz, rock, classical, and more), and get personalised recommendations for similar genres.

## Features

- **Genre classification** – classifies audio into one of 10 genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock
- **Confidence score** – reports how confident the model is in its prediction
- **Genre recommendations** – suggests similar genres with representative artists and a short explanation of why they are similar

## How It Works

1. **Feature extraction** – [librosa](https://librosa.org/) is used to extract audio features: MFCCs (timbral texture), chroma (harmonic content), spectral centroid, spectral roll-off, spectral bandwidth, zero-crossing rate, tempo, and RMS energy.
2. **Classification** – extracted features are compared against representative genre profiles using cosine similarity in a standardised feature space.
3. **Recommendation** – a curated genre-similarity graph maps the detected genre to the most acoustically and culturally related genres.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Classify and get 3 recommendations (default)
python app.py path/to/song.mp3

# Analyse only the first 60 seconds
python app.py path/to/song.wav --duration 60

# Get 5 similar genre recommendations
python app.py path/to/song.flac --top 5

# Analyse the full file
python app.py path/to/song.mp3 --duration 0
```

### Example output

```
🎵  Analysing: path/to/song.mp3
    (using first 30 s of audio)

========================================================
  CLASSIFICATION RESULT
========================================================
  Genre      : JAZZ
  Confidence : [████████████████░░░░] 82.4%
  About      : Jazz – improvisation, complex chords, swing rhythm

  Genre scores (all genres):
    jazz         [████████████░░░] 82.4% ◀
    blues        [██████████░░░░░] 71.3%
    classical    [█████████░░░░░░] 65.1%
    ...

========================================================
  SIMILAR GENRES YOU MIGHT ENJOY
========================================================
  1. BLUES
     Why: Blues is the direct ancestor of jazz
     Artists: B.B. King, Muddy Waters, Robert Johnson, Etta James

  2. CLASSICAL
     Why: Complex harmony and instrumental virtuosity
     Artists: Mozart, Beethoven, Bach, Chopin

  3. REGGAE
     Why: Shares acoustic and cultural roots with jazz
     Artists: Bob Marley, Peter Tosh, Jimmy Cliff, Burning Spear
```

## Supported Formats

Any audio format supported by [libsndfile](http://www.mega-nerd.com/libsndfile/) or [audioread](https://github.com/beetbox/audioread): WAV, MP3, FLAC, OGG, AIFF, and more.

## Project Structure

```
music/
├── app.py                      # CLI entry point
├── requirements.txt
├── music_classifier/
│   ├── __init__.py
│   ├── features.py             # Audio feature extraction
│   ├── classifier.py           # Genre classification
│   └── recommender.py         # Genre recommendation engine
└── tests/
    ├── test_features.py
    ├── test_classifier.py
    └── test_recommender.py
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

