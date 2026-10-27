"""Đọc cấu hình dùng chung của toàn bộ dự án EcoVision AI."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "project.yaml"


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy cấu hình dự án: {CONFIG_PATH}")

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    for section in ("project", "dataset", "training", "inference"):
        if section not in config:
            raise ValueError(f"project.yaml thiếu mục bắt buộc: {section}")
    return config


def project_path(relative_path: str) -> Path:
    """Chuyển đường dẫn tương đối trong project.yaml thành đường dẫn tuyệt đối."""
    return (ROOT_DIR / relative_path).resolve()


CONFIG = _load_config()


def configure_console() -> None:
    """Cho phép CLI in tiếng Việt ổn định trên Windows và Linux."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")
