"""YOLO validation, training, and loss-curve plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


def run_yolo_validation(
    model_name: str = "yolo11n.pt",
    data_yaml: str = "coco8.yaml",
    person_class_id: int = 0,
    imgsz: int = 640,
) -> dict[str, Any]:
    """
    Run YOLO validation and return precision, recall, and mAP metrics.

    Uses the bundled COCO8 dataset (downloads automatically via Ultralytics).
    Filters to the person class for player-detection relevance.
    """
    model = YOLO(model_name)
    results = model.val(
        data=data_yaml,
        classes=[person_class_id],
        imgsz=imgsz,
        verbose=False,
    )

    box = results.box
    return {
        "model_name": model_name,
        "dataset": data_yaml,
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "f1": round(2 * float(box.mp) * float(box.mr) / (float(box.mp) + float(box.mr)), 4)
        if float(box.mp) + float(box.mr) > 0
        else 0.0,
    }


def run_yolo_training(
    model_name: str = "yolo11n.pt",
    data_yaml: str = "coco8.yaml",
    epochs: int = 5,
    imgsz: int = 640,
    project: str = "runs/eval",
    name: str = "yolo_train",
) -> Path:
    """
    Run a short YOLO training session to produce loss curves.

    Returns the path to the run directory containing results.csv.
    """
    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name=name,
        verbose=False,
        plots=True,
    )
    return Path(project) / name


def find_results_csv(run_dir: str | Path) -> Path:
    """Locate results.csv inside an Ultralytics training run directory."""
    run_dir = Path(run_dir)
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found in: {run_dir}")
    return results_csv


def plot_loss_curves(results_csv: str | Path, save_path: str | Path | None = None) -> None:
    """Plot YOLO training and validation loss curves from results.csv."""
    results_csv = Path(results_csv)
    df = pd.read_csv(results_csv)
    df.columns = [col.strip() for col in df.columns]

    loss_columns = [
        ("train/box_loss", "Train Box Loss"),
        ("train/cls_loss", "Train Class Loss"),
        ("train/dfl_loss", "Train DFL Loss"),
        ("val/box_loss", "Val Box Loss"),
        ("val/cls_loss", "Val Class Loss"),
        ("val/dfl_loss", "Val DFL Loss"),
    ]
    available = [(col, label) for col, label in loss_columns if col in df.columns]

    if not available:
        raise ValueError(f"No loss columns found in {results_csv}")

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

    for column, label in available:
        ax.plot(epochs, df[column], label=label, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("YOLO Training & Validation Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_detection_metrics(
    metrics: dict[str, float],
    title: str = "Detection Metrics",
    save_path: str | Path | None = None,
) -> None:
    """Plot precision, recall, F1, and accuracy as a bar chart."""
    label_keys = [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Accuracy", "accuracy"),
        ("mAP@0.5", "map50"),
    ]
    available = [(label, key) for label, key in label_keys if key in metrics]
    labels = [label for label, _ in available]
    values = [metrics[key] for _, key in available]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_precision_recall_curve(threshold_results: list[dict[str, Any]]) -> None:
    """Plot precision and recall across confidence thresholds."""
    thresholds = [row["confidence_threshold"] for row in threshold_results]
    precisions = [row["precision"] for row in threshold_results]
    recalls = [row["recall"] for row in threshold_results]
    f1_scores = [row["f1"] for row in threshold_results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, precisions, marker="o", label="Precision", linewidth=2)
    ax.plot(thresholds, recalls, marker="o", label="Recall", linewidth=2)
    ax.plot(thresholds, f1_scores, marker="o", label="F1", linewidth=2)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Confidence Threshold")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
