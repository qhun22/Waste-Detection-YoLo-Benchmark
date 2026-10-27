"""Tải đúng dataset Roboflow được khai báo trong project.yaml."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from project_config import CONFIG, ROOT_DIR, configure_console


DATASET_CFG = CONFIG["dataset"]


def download(api_key: str, target: Path, force: bool = False) -> None:
    from roboflow import Roboflow

    target = target.resolve()
    if target.exists() and any(target.iterdir()) and not force:
        raise FileExistsError(
            f"Thư mục {target} đã có dữ liệu. Dùng --force nếu chắc chắn muốn thay thế."
        )

    roboflow = Roboflow(api_key=api_key)
    project = roboflow.workspace(DATASET_CFG["roboflow_workspace"]).project(
        DATASET_CFG["roboflow_project"]
    )
    version = project.version(int(DATASET_CFG["version"]))
    print(f"Đang tải {DATASET_CFG['roboflow_project']} v{DATASET_CFG['version']}...")
    dataset = version.download("yolov8")
    source = Path(dataset.location).resolve()

    if source != target:
        if force and target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    print(f"Tải xong: {target}")


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Tải dataset Roboflow cho EcoVision AI")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--target", type=Path, default=ROOT_DIR / "data" / "dataset")
    parser.add_argument("--force", action="store_true", help="Thay thế dataset đang tồn tại")
    args = parser.parse_args()
    download(args.api_key, args.target, args.force)


if __name__ == "__main__":
    main()
