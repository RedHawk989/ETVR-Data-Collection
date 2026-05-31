# This repo is archived as main collection functionality is merged into the main app.

# ETVR-Data-Collection

Tools for collecting and annotating eye tracking data for the LEAPv2 tracking model.

## Structure

- **`collection/`** — Mass data collection GUI (`masscollect.py`). Run `run_collection.bat` on Windows.
- **`annotation/`** — Keypoint annotation (`annotate.py`) and output validation (`validate.py`).

## Usage

### Collection
Run `collection/run_collection.bat` (Windows) or `python collection/masscollect.py` directly. Supports EyeTrackVR cameras and Bigscreen Beyond 2e.

### Annotation
1. Run `python annotation/annotate.py` to extract frames from a video and annotate keypoints.
2. Run `python annotation/validate.py` to visually verify the saved annotations.
