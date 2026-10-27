"""Chạy thử best.pt trên ảnh test và lưu vào results/predictions/test."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from project_config import CONFIG, configure_console, project_path


TRAIN_CFG = CONFIG["training"]
INFERENCE_CFG = CONFIG["inference"]
MODEL_PATH = project_path(TRAIN_CFG["artifact_dir"]) / "weights" / "best.pt"
TEST_IMAGES_DIR = project_path("data/dataset/test/images")
OUTPUT_DIR = project_path("results/predictions")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_inference(n_samples: int = 5, source: str | None = None) -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy trọng số: {MODEL_PATH}")

    if source:
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Không tìm thấy nguồn ảnh: {source_path}")
        prediction_source: str | list[str] = str(source_path)
    else:
        image_paths = sorted(
            str(path) for path in TEST_IMAGES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )[:n_samples]
        if not image_paths:
            raise FileNotFoundError(f"Không tìm thấy ảnh test trong {TEST_IMAGES_DIR}")
        prediction_source = image_paths

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=prediction_source,
        conf=float(INFERENCE_CFG["candidate_confidence"]),
        iou=float(INFERENCE_CFG["iou"]),
        imgsz=int(INFERENCE_CFG["imgsz"]),
        agnostic_nms=True,
        save=True,
        project=str(OUTPUT_DIR),
        name="test",
        exist_ok=True,
    )

    print("=" * 60)
    for result in results:
        print(f"[{Path(result.path).name}]")
        if result.boxes is None or len(result.boxes) == 0:
            print("  Không phát hiện vật thể.")
            continue
        for box in result.boxes:
            class_name = model.names[int(box.cls[0].item())]
            confidence = float(box.conf[0].item())
            print(f"  {class_name:<15} {confidence * 100:.1f}%")

    print(f"Ảnh kết quả: {OUTPUT_DIR / 'test'}")


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Inference ảnh bằng best.pt")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--source", help="Một ảnh, thư mục hoặc video tùy chọn")
    args = parser.parse_args()
    run_inference(n_samples=max(1, args.samples), source=args.source)


if __name__ == "__main__":
    main()
