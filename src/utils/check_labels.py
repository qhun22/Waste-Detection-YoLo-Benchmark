"""Kiểm tra cấu trúc ảnh và nhãn YOLO của train/valid/test."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from project_config import CONFIG, configure_console, project_path


DATA_YAML = project_path(CONFIG["dataset"]["yaml"])
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_split(dataset_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (dataset_dir / path).resolve()


def _validate_label(path: Path, class_count: int) -> tuple[Counter, list[str]]:
    counts: Counter = Counter()
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw_line.split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(f"{path}:{line_number}: cần 5 giá trị")
            continue
        try:
            class_value, x_center, y_center, width, height = map(float, parts)
        except ValueError:
            errors.append(f"{path}:{line_number}: chứa giá trị không phải số")
            continue

        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < class_count:
            errors.append(f"{path}:{line_number}: class id không hợp lệ")
        elif not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            errors.append(f"{path}:{line_number}: tâm bbox ngoài [0, 1]")
        elif not (0 < width <= 1 and 0 < height <= 1):
            errors.append(f"{path}:{line_number}: kích thước bbox không hợp lệ")
        elif x_center - width / 2 < -1e-4 or x_center + width / 2 > 1.0001:
            errors.append(f"{path}:{line_number}: bbox vượt biên ngang")
        elif y_center - height / 2 < -1e-4 or y_center + height / 2 > 1.0001:
            errors.append(f"{path}:{line_number}: bbox vượt biên dọc")
        else:
            counts[class_id] += 1
    return counts, errors


def verify_dataset(yaml_path: Path = DATA_YAML) -> bool:
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy {yaml_path}")

    with yaml_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    names = list(data.get("names", []))
    if not names or int(data.get("nc", len(names))) != len(names):
        raise ValueError("Số lượng names không khớp nc trong data.yaml")

    all_errors: list[str] = []
    total_counts: Counter = Counter()
    print(f"Classes ({len(names)}): {', '.join(names)}")
    for yaml_key, display_name in (("train", "train"), ("val", "valid"), ("test", "test")):
        value = data.get(yaml_key)
        if not value:
            all_errors.append(f"data.yaml thiếu split {yaml_key}")
            continue
        image_dir = _resolve_split(yaml_path.parent, str(value))
        label_dir = image_dir.parent / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            all_errors.append(f"{display_name}: thiếu thư mục ảnh hoặc nhãn")
            continue

        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        if image_stems - label_stems:
            all_errors.append(f"{display_name}: {len(image_stems - label_stems)} ảnh thiếu label")
        if label_stems - image_stems:
            all_errors.append(f"{display_name}: {len(label_stems - image_stems)} label thiếu ảnh")

        split_counts: Counter = Counter()
        for label_path in labels:
            counts, errors = _validate_label(label_path, len(names))
            split_counts.update(counts)
            all_errors.extend(errors)
        total_counts.update(split_counts)
        print(f"{display_name:<5}: {len(images):>5} ảnh | {sum(split_counts.values()):>6} vật thể")

    print("Phân bố toàn bộ nhãn:")
    for class_id, name in enumerate(names):
        print(f"  {name:<15}: {total_counts[class_id]:>6}")

    if all_errors:
        print(f"PHÁT HIỆN {len(all_errors)} LỖI:")
        for error in all_errors[:20]:
            print(f"  - {error}")
        return False

    print("Dataset hợp lệ: ảnh, label, class id và bbox đều đạt kiểm tra.")
    return True


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Kiểm tra nhãn YOLO")
    parser.add_argument("--yaml", type=Path, default=DATA_YAML)
    args = parser.parse_args()
    if not verify_dataset(args.yaml.resolve()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
