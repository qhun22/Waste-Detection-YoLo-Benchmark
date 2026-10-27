"""Đánh giá best.pt trên Test split và cập nhật test_metrics.json."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from ultralytics import YOLO

from project_config import CONFIG, configure_console, project_path


TRAIN_CFG = CONFIG["training"]
DATASET_CFG = CONFIG["dataset"]
MODEL_PATH = project_path(TRAIN_CFG["artifact_dir"]) / "weights" / "best.pt"
DATA_YAML = project_path(DATASET_CFG["yaml"])
OUTPUT_PATH = project_path(TRAIN_CFG["artifact_dir"]) / "test_metrics.json"


def evaluate(device: str, imgsz: int, output: Path) -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Không tìm thấy data.yaml: {DATA_YAML}")

    model = YOLO(str(MODEL_PATH))
    metrics = model.val(
        data=str(DATA_YAML), split="test", imgsz=imgsz,
        device=device, plots=False, verbose=True,
    )

    per_class = []
    for class_id, class_name in model.names.items():
        per_class.append({
            "name": class_name,
            "instances": int(metrics.nt_per_class[class_id]),
            "precision": float(metrics.box.p[class_id]),
            "recall": float(metrics.box.r[class_id]),
            "map50": float(metrics.box.ap50[class_id]),
            "map50_95": float(metrics.box.maps[class_id]),
        })

    result = {
        "split": "Test",
        "evaluated_at": date.today().isoformat(),
        "model": "weights/best.pt",
        "images": int(DATASET_CFG["image_counts"]["test"]),
        "instances": int(sum(metrics.nt_per_class)),
        "settings": {
            "imgsz": imgsz,
            "device": device,
            "ultralytics": __import__("ultralytics").__version__,
        },
        "speed_ms_per_image": {
            "preprocess": float(metrics.speed.get("preprocess", 0.0)),
            "inference": float(metrics.speed.get("inference", 0.0)),
            "postprocess": float(metrics.speed.get("postprocess", 0.0)),
        },
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        },
        "per_class": per_class,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu Test metrics: {output}")
    return result


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Evaluate best.pt trên Test split")
    parser.add_argument("--device", default="cpu", help="cpu hoặc GPU id")
    parser.add_argument("--imgsz", type=int, default=int(TRAIN_CFG["imgsz"]))
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    evaluate(args.device, args.imgsz, args.output.resolve())


if __name__ == "__main__":
    main()
