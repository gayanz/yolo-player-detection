# Sports Player Tracking Pipeline

A computer vision pipeline for detecting, tracking, and estimating body pose of players in sports videos.

**Stack:** Python · PyTorch · YOLO v11 · OpenPose · ByteTrack

---

## Overview

This project processes sports video clips and produces:

- **Player detection** — YOLO v11 (person class)
- **Multi-frame tracking** — ByteTrack (consistent player IDs)
- **Keypoint estimation** — OpenPose (18 body joints per player)

For each video, the pipeline saves an annotated MP4 and a JSON file with per-frame bounding boxes, track IDs, and keypoints.

---

## Video Data

The 6 sports video clips are **not** stored in this repository (to keep the repo lightweight).

Download them from Google Drive:

**[Sports Videos — Google Drive](https://drive.google.com/drive/folders/1TlAWBimU3OzeouoGqYtNQ_4c3YLtDIo8?usp=sharing)**

After downloading, place the `.mkv` files in the project folder:

```
DS5216-PA2/
└── SportsVideos/
    ├── 2025-09-05 12-26-47.mkv
    ├── 2025-09-05 12-31-04.mkv
    ├── 2025-09-05 12-33-35.mkv
    ├── 2025-09-05 12-45-55.mkv
    ├── 2025-09-05 12-53-18.mkv
    └── 2025-09-05 12-56-43.mkv
```

Default paths are already set in `config/config.yaml` (`SportsVideos/` and `outputs/`).

---

## Project Structure

```
DS5216-PA2/
├── config/
│   └── config.yaml              # Paths, model settings, visualization
├── notebooks/
│   └── player_tracking_colab.ipynb   # Google Colab notebook (all steps)
├── scripts/
│   └── run_pipeline.py          # Run pipeline from command line
├── SportsVideos/                # Input videos (download from Google Drive)
├── outputs/                     # Generated results (not committed to git)
├── src/
│   ├── config_loader.py
│   ├── detection/
│   │   └── yolo_detector.py     # YOLO v11 player detection
│   ├── evaluation/
│   │   ├── metrics.py           # Precision, recall, F1, accuracy
│   │   └── yolo_eval.py         # Loss curves and COCO validation
│   ├── keypoints/
│   │   └── openpose_estimator.py # OpenPose keypoints
│   ├── tracking/
│   │   └── player_tracker.py    # ByteTrack multi-object tracking
│   ├── pipeline/
│   │   └── tracking_pipeline.py # End-to-end pipeline
│   └── utils/
│       ├── video_utils.py
│       └── visualization.py
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- CUDA GPU recommended (CPU works but is slower)
- ~2 GB disk space for model weights (downloaded on first run)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Locally

1. Clone the repository and install dependencies (above).
2. Download the videos from Google Drive into `SportsVideos/`.
3. Run the pipeline (paths are already configured in `config/config.yaml`):

```yaml
paths:
  video_dir: "SportsVideos"
  output_dir: "outputs"
```

4. Run the pipeline:

```bash
# Process all videos
python scripts/run_pipeline.py

# Process one video
python scripts/run_pipeline.py --video "SportsVideos/2025-09-05 12-26-47.mkv"

# Quick test (first 50 frames only)
python scripts/run_pipeline.py --max-frames 50
```

### Outputs

Results are written under `outputs/<video_name>/`:

| File | Description |
|------|-------------|
| `*_tracked.mp4` | Annotated video with boxes, IDs, and skeletons |
| `*_results.json` | Per-frame bboxes, track IDs, and keypoints |

---

## Run in Google Colab

1. Open [`notebooks/player_tracking_colab.ipynb`](https://colab.research.google.com/drive/1qcJ1WWD3xkT9IG3Q3YnmRWIOgfIxPSB2?usp=sharing) in Colab.
2. Upload the full `DS5216-PA2` project folder to Google Drive (including `SportsVideos/` and `src/`).
3. Download the 6 videos from the Google Drive link above into `DS5216-PA2/SportsVideos/` if not already there.
4. In **Part 1**, set `PROJECT_ROOT` to your project location on Drive:

```python
PROJECT_ROOT = '/content/drive/MyDrive/DS5216-PA2'
```

Videos and outputs are read from inside the project folder automatically (`SportsVideos/` and `outputs/`).

5. Run all cells top to bottom.

The notebook walks through each stage:

| Part | Step |
|------|------|
| 1 | Setup, install dependencies, load videos |
| 2 | YOLO v11 detection (single frame) |
| 3 | OpenPose keypoints (single player) |
| 4 | ByteTrack tracking (30 frames) |
| 5 | Full pipeline on all videos |
| 6 | Evaluation metrics and loss curves |

> **Note:** Enable a GPU runtime in Colab (`Runtime → Change runtime type → GPU`) for faster processing.

---

## Configuration

Edit [`config/config.yaml`](config/config.yaml) to tune behavior:

| Setting | Default | Description |
|---------|---------|-------------|
| `models.yolo.model_name` | `yolo11n.pt` | YOLO v11 variant (`n`=fastest, `s`/`m`=more accurate) |
| `models.yolo.confidence` | `0.4` | Detection confidence threshold |
| `models.tracking.tracker` | `bytetrack.yaml` | Multi-object tracker |
| `processing.max_frames` | `null` | Limit frames per video (`null` = full video) |
| `processing.save_video` | `true` | Save annotated MP4 |
| `processing.save_json` | `true` | Save JSON results |

---

## Pipeline Flow

```
Video Frame
    │
    ▼
YOLO v11 ──► detect players (person class)
    │
    ▼
ByteTrack ──► assign consistent track IDs
    │
    ▼
OpenPose ──► estimate 18 body keypoints per player crop
    │
    ▼
Annotated video + JSON results
```

---

## License

Academic / course project — DS5216 PA2.
