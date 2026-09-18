import argparse
import os
import shutil
from ultralytics import YOLO

def train_model(model_name, epochs, batch_size, imgsz, device):
    data_yaml = os.path.abspath("data/dataset/data.yaml")
    
    print(f"\n{'='*50}")
    print(f"Bắt đầu huấn luyện mô hình: {model_name}")
    print(f"Cấu hình: Epochs={epochs}, Batch={batch_size}, Image Size={imgsz}, Device={device}")
    print(f"{'='*50}\n")

    model = YOLO(model_name)

    project_dir = "runs/train"
    experiment_name = model_name.replace(".pt", "")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project_dir,
        name=experiment_name,
        save=True,
        plots=True
    )

    weights_dir = "results/weights"
    os.makedirs(weights_dir, exist_ok=True)
    best_weight_src = os.path.join(project_dir, experiment_name, "weights", "best.pt")
    best_weight_dst = os.path.join(weights_dir, f"{experiment_name}_best.pt")

    if os.path.exists(best_weight_src):
        shutil.copy(best_weight_src, best_weight_dst)
        print(f"\n=> Đã lưu checkpoint tối ưu tại: {best_weight_dst}")

    print(f"\nHoàn thành huấn luyện mô hình: {model_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO models on Garbage Dataset")
    parser.add_argument("--model", type=str, default="yolov8n.pt", 
                        help="Tên mô hình: yolov8n.pt, yolov8s.pt, yolo11n.pt, yolov5su.pt")
    parser.add_argument("--epochs", type=int, default=50, help="Số lượng epoch")
    parser.add_argument("--batch", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--device", type=str, default="0", 
                        help="GPU device id (ví dụ: '0') hoặc 'cpu'")

    args = parser.parse_args()
    train_model(args.model, args.epochs, args.batch, args.imgsz, args.device)