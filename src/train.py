"""Huấn luyện YOLO11n bằng cấu hình chuẩn trong project.yaml."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from project_config import CONFIG, configure_console, project_path


TRAIN_CFG = CONFIG["training"]
DATA_YAML = project_path(CONFIG["dataset"]["yaml"])


def train(
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    run_name: str,
    exist_ok: bool,
) -> Path:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset YAML: {DATA_YAML}")

    train_project = project_path(TRAIN_CFG["train_project"])
    artifact_dir = project_path(TRAIN_CFG["artifact_dir"])
    print(
        f"Train YOLO11n | epochs={epochs} batch={batch} imgsz={imgsz} "
        f"device={device} run={run_name}"
    )

    model = YOLO(TRAIN_CFG["model"])
    model.train(
        data=str(DATA_YAML), epochs=epochs, batch=batch, imgsz=imgsz,
        device=device, workers=workers, project=str(train_project),
        name=run_name, exist_ok=exist_ok,
        pretrained=bool(TRAIN_CFG["pretrained"]),
        optimizer=TRAIN_CFG["optimizer"], seed=int(TRAIN_CFG["seed"]),
        deterministic=bool(TRAIN_CFG["deterministic"]),
        amp=bool(TRAIN_CFG["amp"]), close_mosaic=int(TRAIN_CFG["close_mosaic"]),
        save=True, plots=True,
    )

    run_dir = Path(model.trainer.save_dir).resolve()
    best_path = run_dir / "weights" / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(f"Train hoàn tất nhưng không tìm thấy best.pt tại {best_path}")

    if run_name == TRAIN_CFG["run_name"] and run_dir != artifact_dir:
        artifact_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run_dir, artifact_dir, dirs_exist_ok=True)
        published_best = artifact_dir / "weights" / "best.pt"
    else:
        published_best = best_path

    print(f"Checkpoint tốt nhất: {published_best}")
    return published_best


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Train YOLO11n cho EcoVision AI")
    parser.add_argument("--epochs", type=int, default=int(TRAIN_CFG["epochs"]))
    parser.add_argument("--batch", type=int, default=int(TRAIN_CFG["batch"]))
    parser.add_argument("--imgsz", type=int, default=int(TRAIN_CFG["imgsz"]))
    parser.add_argument("--device", default=str(TRAIN_CFG["device"]), help="GPU id hoặc cpu")
    parser.add_argument("--workers", type=int, default=int(TRAIN_CFG["workers"]))
    parser.add_argument("--run-name", default=TRAIN_CFG["run_name"])
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
