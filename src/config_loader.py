"""Configuration loader for the player tracking pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
PROJECT_ROOT = DEFAULT_CONFIG_PATH.parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and resolve paths relative to the project root."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for key in ("video_dir", "output_dir"):
        resolved = Path(config["paths"][key])
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        config["paths"][key] = str(resolved)

    return config


def get_video_paths(config: dict[str, Any]) -> list[Path]:
    """Return sorted video files from the configured directory."""
    video_dir = Path(config["paths"]["video_dir"])
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted(
        path for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
