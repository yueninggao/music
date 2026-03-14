# Music-Video Recommendation System

An end-to-end pipeline that classifies a video and recommends matching music,
then produces a complete creative package for editors and music supervisors.

## Pipeline stages

```
Video Input  ──►  VideoProcessor  ──►  VideoClassifier  ──►  MusicRecommender  ──►  CreativePipeline
  (INPUT)         frame extraction     scene / mood /         ranked track list      playlist · tempo map
                  & feature stats      activity / pace        with explanations      mood arc · brief
```

### Stage 1 – INPUT (`src/video_processor.py`)
Loads a video file (or a deterministic synthetic source for testing), extracts
frames at a configurable interval, and computes per-frame visual features:
brightness, colour histogram, edge density, and motion score.  All features
are aggregated into a `FrameBundle`.

Real video files require OpenCV (`pip install opencv-python`); the synthetic
backend works without any extra dependencies.

### Stage 2 – RECOGNITION & CLASSIFICATION (`src/classifier.py`)
Applies a rule-based decision matrix to the `FrameBundle` aggregate statistics
to produce a `ClassificationResult` with ranked label lists for:

| Dimension | Example labels |
|-----------|---------------|
| Scene     | concert · nature · urban · beach · indoor · night … |
| Activity  | dancing · sports · relaxing · driving · celebration … |
| Mood      | energetic · calm · melancholic · happy · tense · romantic … |
| Pace      | slow · moderate · fast |
| Lighting  | dark · dim · normal · bright |

An optional `model_fn` hook lets you inject a real ML classifier (e.g. CLIP)
whose predictions are fused (with 2× weight) with the rule-based scores.

### Stage 3 – DECISION-MAKING & RECOMMENDATION (`src/recommender.py`)
Scores every track in the built-in 20-track cross-genre catalogue against the
`ClassificationResult` using a weighted matrix:

| Attribute    | Weight |
|--------------|--------|
| Mood match   | 35 %   |
| Activity     | 25 %   |
| Scene        | 20 %   |
| Pace / BPM   | 15 %   |
| Energy level |  5 %   |

Returns a ranked `RecommendationOutput` with explainability reasons per track.
Accepts a custom catalogue to replace or extend the built-in library.

### Stage 4 – CREATIVE STEPS (`src/creative_pipeline.py`)
Four creative post-processing sub-steps produce the final `CreativePackage`:

1. **PlaylistCurator** – orders tracks into an energy arc (low → peak → cool-down)
   and adds musical transition descriptors between each pair.
2. **TempoMapper** – divides the video duration into labelled segments (intro /
   build / climax / resolution / outro) and assigns a BPM target to each.
3. **MoodArcBuilder** – generates a mood timeline with interpolated intensity
   keyframes for the full video duration.
4. **CreativeNoteWriter** – composes a human-readable creative brief with
   playlist narrative, sync notes, and creative direction for editors.

## Quick start

```bash
# No extra dependencies needed (pure-Python synthetic demo)
python main.py --source "synthetic:60"

# With a real video (requires: pip install opencv-python)
python main.py --source /path/to/video.mp4 --opencv

# JSON output, top 3 recommendations
python main.py --source "synthetic:90" --top-n 3 --json

# Custom frame sampling interval
python main.py --source "synthetic:120" --interval 2.0
```

## Project layout

```
music/
├── main.py                        CLI entry-point
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── video_processor.py         Stage 1 – INPUT
│   ├── classifier.py              Stage 2 – RECOGNITION & CLASSIFICATION
│   ├── recommender.py             Stage 3 – DECISION-MAKING & RECOMMENDATION
│   ├── creative_pipeline.py       Stage 4 – CREATIVE STEPS
│   └── pipeline.py                End-to-end orchestrator
└── tests/
    ├── test_video_processor.py
    ├── test_classifier.py
    ├── test_recommender.py
    ├── test_creative_pipeline.py
    └── test_pipeline.py
```

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

137 tests covering every module and edge case.
