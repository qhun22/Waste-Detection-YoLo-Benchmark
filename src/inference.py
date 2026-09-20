import os
import glob
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "results/weights/yolov8s_best.pt"
TEST_IMAGES_DIR = "data/dataset/test/images"
OUTPUT_DIR = "results/predictions"

WASTE_CATEGORY_MAPPING = {
    "BIODEGRADABLE": "Rác Hữu Cơ",
    "CARDBOARD": "Rác Tái Chế / Vô Cơ",
    "GLASS": "Rác Tái Chế / Vô Cơ",
    "METAL": "Rác Tái Chế / Vô Cơ",
    "PAPER": "Rác Tái Chế / Vô Cơ",
    "PLASTIC": "Rác Tái Chế / Vô Cơ",
}

def run_inference():
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file trọng số tại {MODEL_PATH}")
        return

    print(f"--> Khởi tạo mô hình: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))[:5]
    if not image_paths:
        print("Lỗi: Không tìm thấy ảnh nào trong thư mục test!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--> Đang chạy nhận diện trên {len(image_paths)} ảnh mẫu...")

    results = model.predict(
        source=image_paths,
        conf=0.25,
        save=True,
        save_dir=OUTPUT_DIR,
        project="results/predictions",
        name="",
        exist_ok=True
    )

    print("\n" + "="*50)
    print("KẾT QUẢ PHÂN LOẠI CHI TIẾT:")
    print("="*50)

    for res in results:
        img_name = Path(res.path).name
        print(f"\n[Ảnh: {img_name}]")
        boxes = res.boxes
        if len(boxes) == 0:
            print("  - Không phát hiện vật thể rác nào.")
            continue

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            cls_name = model.names[cls_id]
            category = WASTE_CATEGORY_MAPPING.get(cls_name, "Không xác định")
            print(f"  + Phát hiện: {cls_name:<15} | Độ tin cậy: {conf_val*100:.1f}% | Phân loại: {category}")

    print(f"\n--> Ảnh kết quả đóng bounding box đã được lưu tại: {OUTPUT_DIR}/")

if __name__ == "__main__":
    run_inference()