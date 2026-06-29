"""Configuration loader for the player tracking pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and resolve common paths."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["paths"]["video_dir"] = str(Path(config["paths"]["video_dir"]))
    config["paths"]["output_dir"] = str(Path(config["paths"]["output_dir"]))
    return config


def get_video_paths(config: dict[str, Any]) -> list[Path]:
    """Return sorted video files from the configured directory."""
    video_dir = Path(config["paths"]["video_dir"])
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted(
        path for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
