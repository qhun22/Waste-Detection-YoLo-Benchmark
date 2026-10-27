"""
EcoVision AI — FastAPI Backend
Phân loại rác thải thời gian thực bằng YOLO trên CPU.

Cài đặt:
    pip install fastapi uvicorn[standard] python-multipart jinja2 opencv-python pillow ultralytics

Khởi động:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import csv
import os
import io
import base64
import json
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from project_config import CONFIG, project_path

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# ─────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(BASE_DIR)
DATASET_CFG   = CONFIG["dataset"]
TRAIN_CFG     = CONFIG["training"]
INFERENCE_CFG = CONFIG["inference"]

TRAIN_RUN_DIR = os.getenv(
    "ECOVISION_ARTIFACT_DIR",
    str(project_path(TRAIN_CFG["artifact_dir"])),
)
MODEL_PATH = os.getenv(
    "ECOVISION_MODEL_PATH",
    os.path.join(TRAIN_RUN_DIR, "weights", "best.pt"),
)
OPENVINO_DIR  = os.path.join(TRAIN_RUN_DIR, "weights", "best_openvino_model")
RESULTS_CSV   = os.path.join(TRAIN_RUN_DIR, "results.csv")
TEST_METRICS_JSON = os.path.join(TRAIN_RUN_DIR, "test_metrics.json")

DATASET_NAME     = DATASET_CFG["name"]
DATASET_VERSION  = DATASET_CFG["version"]
DATASET_DISPLAY_NAME = DATASET_CFG.get(
    "display_name",
    f"{DATASET_NAME}_v{DATASET_VERSION}",
)
TRAIN_RUN_NAME   = TRAIN_CFG["run_name"]
TRAINING_DEVICE  = TRAIN_CFG["device_name"]

CAM_W         = 640
CAM_H         = 480
CAM_IMGSZ     = int(INFERENCE_CFG["imgsz"])
CAM_CONF_DEF  = float(INFERENCE_CFG["webcam_confidence"])
CAM_MAX_DET   = 5
CAM_BOX_RATIO = 0.60   # Loại box chiếm > 60% diện tích frame

IMG_IMGSZ = int(INFERENCE_CFG["imgsz"])
IMG_CONF  = float(INFERENCE_CFG["candidate_confidence"])
IMG_IOU   = float(INFERENCE_CFG["iou"])
IMG_REVIEW_CONF = float(INFERENCE_CFG["confirmed_confidence"])
IMG_BOX_RATIO   = 0.97  # Loại box bất thường gần phủ kín toàn bộ ảnh

VIDEO_IMGSZ       = int(INFERENCE_CFG["imgsz"])
VIDEO_CONF        = float(INFERENCE_CFG["candidate_confidence"])
VIDEO_IOU         = float(INFERENCE_CFG["iou"])
VIDEO_MAX_DET     = 20
VIDEO_BOX_RATIO   = 0.97   # vật chụp cận cảnh, gần kín khung hình, vẫn là detection hợp lệ
VIDEO_INFERENCE_FPS = 15.0 # Lấy mẫu tối đa 15 FPS, tái dùng box giữa các frame liền kề
VIDEO_UPLOAD_DIR  = os.path.join(BASE_DIR, "static", "videos", "uploads")
VIDEO_OUTPUT_DIR  = os.path.join(BASE_DIR, "static", "videos", "outputs")
VIDEO_MAX_BYTES   = 100 * 1024 * 1024   # 100 MB
VIDEO_ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

_video_jobs_lock = threading.Lock()
_video_jobs: dict = {}   # job_id -> {status, progress, error, original_url, result_url}
_video_worker_semaphore = threading.Semaphore(1)


# ─────────────────────────────────────────────────────────────────────
#  Thread-safe Shared State
# ─────────────────────────────────────────────────────────────────────
_raw_lock   = threading.Lock()
_latest_raw: np.ndarray | None = None   # BGR frame từ OpenCV

_ann_lock   = threading.Lock()
_latest_ann: np.ndarray | None = None   # BGR frame đã annotate

_stats_lock = threading.Lock()
_stats: dict = {"fps": 0.0, "detections": 0, "conf": CAM_CONF_DEF}

_camera_enabled = threading.Event()

# ─────────────────────────────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────────────────────────────
def _load_model() -> YOLO:
    """Ưu tiên OpenVINO nếu có export, fallback sang PyTorch."""
    if os.path.isdir(OPENVINO_DIR):
        print(f"[EcoVision] OpenVINO model: {OPENVINO_DIR}")
        return YOLO(OPENVINO_DIR)
    print(f"[EcoVision] PyTorch model: {MODEL_PATH}")
    return YOLO(MODEL_PATH)

print("[EcoVision] Loading YOLO11n...")
model = _load_model()
_model_lock = threading.Lock()
print("[EcoVision] Model ready.")

# ─────────────────────────────────────────────────────────────────────
#  Helper: Lọc bounding box bất thường
# ─────────────────────────────────────────────────────────────────────
def _filter_boxes(result, frame_area: int, box_ratio: float = CAM_BOX_RATIO) -> None:
    """
    Lọc in-place:
      1. Box diện tích > box_ratio → background / toàn thân người
      2. Aspect ratio cực đoan → tay/chân người dọc theo cạnh frame

    box_ratio mặc định (CAM_BOX_RATIO) chỉ hợp lý cho webcam, nơi khung hình
    luôn có nền + người đứng cầm vật. Với video upload, một vật thể chụp cận
    cảnh chiếm gần hết khung hình là chuyện bình thường — dùng ngưỡng cao hơn
    (VIDEO_BOX_RATIO) để không loại nhầm detection đúng.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return
    xyxy   = result.boxes.xyxy
    w      = xyxy[:, 2] - xyxy[:, 0]
    h      = xyxy[:, 3] - xyxy[:, 1]
    areas  = w * h
    aspect = w / (h + 1e-6)
    keep   = (
        (areas / frame_area <= box_ratio) &
        (aspect >= 0.12) &
        (aspect <= 7.0)
    )
    result.boxes = Boxes(result.boxes.data[keep], result.boxes.orig_shape)


def _draw_dashed_rectangle(
    frame: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash: int = 8,
) -> None:
    """Vẽ bbox nét đứt để phân biệt dự đoán chưa đạt ngưỡng xác nhận."""
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def _draw_group_labels(
    frame: np.ndarray,
    result,
    review_conf: float | None = None,
) -> np.ndarray:
    """Vẽ bbox; dự đoán dưới review_conf dùng nét đứt và dấu hỏi."""
    # OpenCV expects BGR tuples. These match the class colours used by the UI.
    color_map = {
        "BIODEGRADABLE": (94, 197,  34),   # #22c55e green
        "GLASS":         (247, 85, 168),   # #a855f7 purple
        "METAL":         (8,  179, 234),   # #eab308 yellow
        "PAPER":         (22, 115, 249),   # #f97316 orange
        "PLASTIC":       (68,  68, 239),   # #ef4444 red
    }

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        label  = model.names[cls_id]
        conf   = float(box.conf[0].item())
        uncertain = review_conf is not None and conf < review_conf
        display_text = f"{'? ' if uncertain else ''}{label} {conf * 100:.0f}%"
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_color = color_map.get(label, (150, 150, 150))

        if uncertain:
            _draw_dashed_rectangle(frame, (x1, y1), (x2, y2), box_color)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        (text_width, text_height), _ = cv2.getTextSize(
            display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        top = max(0, y1 - text_height - 10)
        bottom = max(text_height + 6, y1)
        cv2.rectangle(frame, (x1, top), (x1 + text_width + 8, bottom), box_color, -1)
        cv2.putText(
            frame,
            display_text,
            (x1 + 4, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame

# ─────────────────────────────────────────────────────────────────────
#  Background Thread 1: Camera Capture
# ─────────────────────────────────────────────────────────────────────
def _camera_worker() -> None:
    """
    Đọc frame từ webcam liên tục.
    CAP_PROP_BUFFERSIZE=1 đảm bảo luôn lấy frame MỚI NHẤT, tránh lag buffer.
    Pipeline màu: cv2 trả về BGR → lưu nguyên vào _latest_raw.
    """
    global _latest_raw
    cap = None
    while True:
        if not _camera_enabled.is_set():
            if cap is not None:
                cap.release()
                cap = None
                print("[EcoVision] Camera stopped.")
            with _raw_lock:
                _latest_raw = None
            time.sleep(0.1)
            continue

        if cap is None:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            cap.set(cv2.CAP_PROP_FPS,          30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            if not cap.isOpened():
                cap.release()
                cap = None
                print("[EcoVision] ERROR: Cannot open webcam (index 0)!")
                time.sleep(1)
                continue
            print("[EcoVision] Camera started.")

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        with _raw_lock:
            _latest_raw = frame  # BGR numpy array

# ─────────────────────────────────────────────────────────────────────
#  Background Thread 2: YOLO Inference
# ─────────────────────────────────────────────────────────────────────
def _inference_worker() -> None:
    """
    Đọc frame MỚI NHẤT → chạy YOLO → ghi kết quả.
    Non-blocking với MJPEG generator và HTTP handlers.

    Pipeline màu:
      _latest_raw (BGR) → model.predict(source=BGR) → result.plot() (BGR)
      → lưu vào _latest_ann → cv2.imencode (encode đúng màu cho browser)
    """
    global _latest_ann
    fps_count = 0
    fps_t0    = time.perf_counter()

    print("[EcoVision] Inference thread started.")
    while True:
        if not _camera_enabled.is_set():
            with _ann_lock:
                _latest_ann = None
            with _stats_lock:
                _stats["fps"] = 0.0
                _stats["detections"] = 0
            time.sleep(0.1)
            continue

        with _raw_lock:
            frame = _latest_raw
        if frame is None:
            time.sleep(0.033)
            continue

        fh, fw     = frame.shape[:2]
        frame_area = fw * fh

        with _stats_lock:
            conf = _stats["conf"]

        try:
            with _model_lock:
                result = model.predict(
                    source=frame, conf=conf, imgsz=CAM_IMGSZ,
                    max_det=CAM_MAX_DET, agnostic_nms=True, verbose=False,
                )[0]
            _filter_boxes(result, frame_area)

            annotated = frame.copy()
            n_det = len(result.boxes) if result.boxes is not None else 0
            if n_det:
                annotated = _draw_group_labels(annotated, result)

            with _ann_lock:
                _latest_ann = annotated

            # Cập nhật FPS stats mỗi giây
            fps_count += 1
            elapsed = time.perf_counter() - fps_t0
            if elapsed >= 1.0:
                with _stats_lock:
                    _stats["fps"]        = round(fps_count / elapsed, 1)
                    _stats["detections"] = n_det
                fps_count = 0
                fps_t0    = time.perf_counter()

        except Exception as exc:
            print(f"[EcoVision] Inference error: {exc}")
            time.sleep(0.05)

def _mjpeg_gen():
    """
    Yields MJPEG boundary-separated JPEG frames ở tối đa 30 FPS.
    Màu sắc: BGR annotated → cv2.imencode → browser hiển thị chuẩn RGB.
    (OpenCV JPEG encoder tự xử lý BGR→display conversion.)
    """
    while True:
        if not _camera_enabled.is_set():
            break

        with _ann_lock:
            frame = _latest_ann

        # Không gửi placeholder màu đen. Trình duyệt sẽ giữ overlay "Đang kết nối"
        # cho tới khi webcam và model thực sự tạo được frame đầu tiên.
        if frame is None:
            time.sleep(0.05)
            continue

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
        time.sleep(1 / 30)  # cap output tại 30 FPS

# ─────────────────────────────────────────────────────────────────────
#  FastAPI Application
# ─────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    for target, name in [
        (_camera_worker,    "ecovision-cam"),
        (_inference_worker, "ecovision-infer"),
    ]:
        threading.Thread(target=target, daemon=True, name=name).start()
    yield

app = FastAPI(title="EcoVision AI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ── Test-image endpoint (dev only) ────────────────────────────
_TEST_IMG_DIR = os.path.join(ROOT_DIR, "results", "test_images")
if os.path.isdir(_TEST_IMG_DIR):
    app.mount("/test-images", StaticFiles(directory=_TEST_IMG_DIR), name="test-images")

# ── Training artifacts (biểu đồ, ma trận nhầm lẫn...) ──────────
if os.path.isdir(TRAIN_RUN_DIR):
    app.mount("/train-artifacts", StaticFiles(directory=TRAIN_RUN_DIR), name="train-artifacts")

DATASET_CLASSES = list(DATASET_CFG["classes"])


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/video_feed", summary="MJPEG webcam stream")
async def video_feed():
    return StreamingResponse(
        _mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/cam-stats")
async def cam_stats():
    with _stats_lock:
        return JSONResponse({**_stats, "enabled": _camera_enabled.is_set()})


@app.post("/api/camera-toggle")
async def camera_toggle(enabled: bool):
    global _latest_raw, _latest_ann

    if enabled:
        _camera_enabled.set()
    else:
        _camera_enabled.clear()
        with _raw_lock:
            _latest_raw = None
        with _ann_lock:
            _latest_ann = None
        with _stats_lock:
            _stats["fps"] = 0.0
            _stats["detections"] = 0

    return JSONResponse({"status": "ok", "enabled": _camera_enabled.is_set()})


@app.get("/api/cam-debug")
async def cam_debug():
    """Raw inference on latest frame — no _filter_boxes, returns box details."""
    with _raw_lock:
        frame = _latest_raw
    if frame is None:
        return JSONResponse({"error": "no frame"})
    fh, fw = frame.shape[:2]
    frame_area = fw * fh
    with _model_lock:
        res = model.predict(source=frame, conf=0.20, imgsz=CAM_IMGSZ,
                            max_det=20, agnostic_nms=True, verbose=False)[0]
    boxes = []
    for b in (res.boxes or []):
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        w, h = x2-x1, y2-y1
        ratio = (w*h) / frame_area
        boxes.append({
            "label": model.names[int(b.cls[0])],
            "conf": round(float(b.conf[0])*100, 1),
            "box": [x1,y1,x2,y2],
            "area_ratio": round(ratio, 3),
            "kept": ratio <= CAM_BOX_RATIO,
        })
    return JSONResponse({"frame": f"{fw}x{fh}", "raw_count": len(boxes), "boxes": boxes})


@app.post("/api/set-conf")
async def set_conf(conf: float = CAM_CONF_DEF):
    val = round(max(0.10, min(0.90, conf)), 2)
    with _stats_lock:
        _stats["conf"] = val
    return JSONResponse({"status": "ok", "conf": val})


def _runtime_device_info() -> dict:
    """Thiết bị thực sự được ứng dụng dùng trên máy đang chạy server."""
    if os.path.isdir(OPENVINO_DIR):
        return {"type": "CPU", "name": "CPU", "engine": "OpenVINO"}

    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        return {
            "type": "GPU",
            "name": torch.cuda.get_device_name(index),
            "engine": "CUDA",
        }

    return {
        "type": "CPU",
        "name": "CPU",
        "engine": "PyTorch",
    }


@app.get("/api/training-stats")
async def training_stats():
    """Số liệu huấn luyện thật, đọc trực tiếp từ results.csv của lần train PhanLoaiRac_5Class_v2."""
    if not os.path.exists(RESULTS_CSV):
        return JSONResponse({"error": "Không tìm thấy results.csv"}, status_code=404)

    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return JSONResponse({"error": "results.csv rỗng"}, status_code=404)

    def _f(row, key):
        try:
            return float(row[key])
        except (KeyError, ValueError):
            return None

    epochs = []
    for r in rows:
        epochs.append({
            "epoch":      int(float(r["epoch"])),
            "precision":  _f(r, "metrics/precision(B)"),
            "recall":     _f(r, "metrics/recall(B)"),
            "map50":      _f(r, "metrics/mAP50(B)"),
            "map50_95":   _f(r, "metrics/mAP50-95(B)"),
            "box_loss":   _f(r, "train/box_loss"),
            "cls_loss":   _f(r, "train/cls_loss"),
            "dfl_loss":   _f(r, "train/dfl_loss"),
        })

    best = max(epochs, key=lambda e: (e["map50_95"] or -1))
    last = epochs[-1]

    test_metrics = None
    if os.path.exists(TEST_METRICS_JSON):
        try:
            with open(TEST_METRICS_JSON, encoding="utf-8") as f:
                test_metrics = json.load(f)
        except (OSError, json.JSONDecodeError):
            test_metrics = None

    return JSONResponse({
        "model":        "YOLO11n",
        "dataset":      DATASET_DISPLAY_NAME,
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "training_run": TRAIN_RUN_NAME,
        "training_device": TRAINING_DEVICE,
        "runtime_device": _runtime_device_info(),
        "metrics_split": "Validation",
        "classes":      DATASET_CLASSES,
        "num_classes":  len(DATASET_CLASSES),
        "total_epochs": len(epochs),
        "best_epoch":   best,
        "last_epoch":   last,
        "history":      epochs,
        "dataset_counts": DATASET_CFG["image_counts"],
        "label_counts": DATASET_CFG["label_counts"],
        "label_counts_scope": "Train + Validation + Test",
        "labels_artifact_scope": "Train",
        "test_metrics": test_metrics,
    })


@app.get("/api/training-artifact/{name}")
async def training_artifact(name: str):
    """Trả về trực tiếp 1 file ảnh kết quả huấn luyện (confusion_matrix.png, results.png, ...)."""
    allowed = {
        "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
        "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
        "labels.jpg",
    }
    if name not in allowed:
        raise HTTPException(status_code=404, detail="Không hỗ trợ file này.")
    path = os.path.join(TRAIN_RUN_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File chưa tồn tại.")
    return FileResponse(path)


@app.post("/api/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        data    = await file.read()
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File ảnh không hợp lệ.")

    t0 = time.perf_counter()
    with _model_lock:
        res = model.predict(
            source=pil_img,
            conf=IMG_CONF,
            iou=IMG_IOU,
            imgsz=IMG_IMGSZ,
            max_det=100,
            agnostic_nms=True,
            verbose=False,
        )[0]
    total_ms = (time.perf_counter() - t0) * 1000

    img_w, img_h = pil_img.size
    frame_area   = img_w * img_h
    _filter_boxes(res, frame_area, IMG_BOX_RATIO)

    detections: list[dict] = []
    if res.boxes and len(res.boxes) > 0:
        for box in res.boxes:
            cls_id   = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            label    = model.names[cls_id]
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            box_area = (x2 - x1) * (y2 - y1)
            detections.append({
                "label":      label,
                "conf":       round(conf_val * 100, 1),
                "box":        [round(x1), round(y1), round(x2), round(y2)],
                "area_ratio": round(box_area / frame_area * 100, 1),
                "status":     "confirmed" if conf_val >= IMG_REVIEW_CONF else "uncertain",
            })

    # Giữ ứng viên yếu trên ảnh bằng nét đứt/dấu hỏi. UI chỉ tính box đạt
    # IMG_REVIEW_CONF là kết quả xác nhận.
    img_bgr       = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    annotated_bgr = _draw_group_labels(img_bgr, res, review_conf=IMG_REVIEW_CONF)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    out_pil       = Image.fromarray(annotated_rgb)

    buf = io.BytesIO()
    out_pil.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    confirmed_count = sum(det["status"] == "confirmed" for det in detections)
    uncertain_count = len(detections) - confirmed_count

    return JSONResponse({
        "image_b64":   img_b64,
        "detections":  detections,
        "count":       confirmed_count,
        "candidate_count": len(detections),
        "uncertain_count": uncertain_count,
        "review_conf": round(IMG_REVIEW_CONF * 100, 1),
        "inference_size": IMG_IMGSZ,
        "image_size":  {"width": img_w, "height": img_h},
        "timing_ms": {
            "preprocess":  round(res.speed.get("preprocess", 0.0), 1),
            "inference":   round(res.speed.get("inference", 0.0), 1),
            "postprocess": round(res.speed.get("postprocess", 0.0), 1),
            "total":       round(total_ms, 1),
        },
    })


# ─────────────────────────────────────────────────────────────────────
#  Video Upload — Xử lý nhận diện từng frame, ghi ra video annotated
# ─────────────────────────────────────────────────────────────────────
def _transcode_for_browser(input_path: str, playable_path: str, can_remux: bool = False) -> bool:
    """Chuẩn hóa video gốc cho trình duyệt; remux H.264 khi có thể để tránh encode lại."""
    codec_args = (
        ["-c:v", "copy"]
        if can_remux
        else ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast"]
    )
    r = subprocess.run(
        [
            FFMPEG_EXE, "-y", "-i", input_path,
            "-an", *codec_args, "-movflags", "+faststart",
            playable_path,
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0 and os.path.exists(playable_path)


def _process_video_job(job_id: str, input_path: str, playable_path: str, output_path: str) -> None:
    def _set(**kwargs):
        with _video_jobs_lock:
            _video_jobs[job_id].update(kwargs)

    process_t0 = time.perf_counter()

    with _video_jobs_lock:
        if _video_jobs.get(job_id, {}).get("cancel_requested"):
            _set(status="cancelled", phase="cancelled", error="Đã hủy do chọn video mới.")
            return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        _set(status="error", error="Không thể đọc file video.")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_s   = total_frames / fps if total_frames > 0 and fps > 0 else 0.0
    inference_stride = max(1, round(fps / VIDEO_INFERENCE_FPS))
    fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec        = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4)).strip().lower()
    input_ext    = os.path.splitext(input_path)[1].lower()
    can_remux    = input_ext in {".mp4", ".mov"} and codec in {"avc1", "h264", "x264"}

    _set(
        phase="preparing",
        width=width,
        height=height,
        fps=round(fps, 2),
        total_frames=total_frames,
        duration_s=round(duration_s, 2),
        source_codec=codec.upper() or "UNKNOWN",
        source_mode="remux" if can_remux else "transcode",
        inference_stride=inference_stride,
    )

    if not _transcode_for_browser(input_path, playable_path, can_remux=can_remux):
        cap.release()
        _set(status="error", error="Không thể đọc hoặc chuyển đổi file video gốc.")
        return

    _set(phase="processing")

    # Encode qua ffmpeg (libx264) để trình duyệt phát được trực tiếp —
    # cv2.VideoWriter không có encoder H.264 khả dụng trên build OpenCV mặc định.
    ffmpeg_proc = subprocess.Popen(
        [
            FFMPEG_EXE, "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "ultrafast", "-movflags", "+faststart",
            output_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frame_idx = 0
    frames_with_detections = 0
    analyzed_frames = 0
    detection_events = 0
    confidence_sum = 0.0
    class_counts: dict[str, int] = {}
    latest_result = None
    try:
        while True:
            with _video_jobs_lock:
                cancelled = _video_jobs.get(job_id, {}).get("cancel_requested", False)
            if cancelled:
                _set(status="cancelled", phase="cancelled", error="Đã hủy do chọn video mới.")
                return
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if (frame_idx - 1) % inference_stride == 0:
                with _model_lock:
                    result = model.predict(
                        source=frame, conf=VIDEO_CONF, iou=VIDEO_IOU, imgsz=VIDEO_IMGSZ,
                        max_det=VIDEO_MAX_DET, agnostic_nms=True, verbose=False,
                    )[0]
                frame_area = width * height
                _filter_boxes(result, frame_area, box_ratio=VIDEO_BOX_RATIO)
                latest_result = result
                analyzed_frames += 1

                if result.boxes is not None and len(result.boxes) > 0:
                    frames_with_detections += 1
                    detection_events += len(result.boxes)
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        label = model.names[cls_id]
                        class_counts[label] = class_counts.get(label, 0) + 1
                        confidence_sum += float(box.conf[0].item())

            annotated = frame
            if latest_result is not None and latest_result.boxes is not None and len(latest_result.boxes) > 0:
                annotated = _draw_group_labels(frame, latest_result)

            ffmpeg_proc.stdin.write(annotated.tobytes())

            if total_frames > 0 and (frame_idx % 10 == 0 or frame_idx == total_frames):
                progress = round(frame_idx / total_frames * 100, 1)
                _set(progress=min(progress, 99.9), processed_frames=frame_idx)
        _set(phase="finalizing", processed_frames=frame_idx)
    except Exception as exc:
        _set(status="error", error=f"Lỗi xử lý video: {exc}")
        return
    finally:
        cap.release()
        with _video_jobs_lock:
            cancelled = _video_jobs.get(job_id, {}).get("cancel_requested", False)
        if cancelled and ffmpeg_proc.poll() is None:
            ffmpeg_proc.terminate()
        if ffmpeg_proc.stdin and not ffmpeg_proc.stdin.closed:
            ffmpeg_proc.stdin.close()
        try:
            ffmpeg_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ffmpeg_proc.kill()
            ffmpeg_proc.wait()

    if ffmpeg_proc.returncode != 0 or not os.path.exists(output_path):
        _set(status="error", error="Không thể tạo file video đầu ra.")
        return

    processing_ms = (time.perf_counter() - process_t0) * 1000
    _set(
        status="done",
        phase="done",
        progress=100.0,
        processed_frames=frame_idx,
        analyzed_frames=analyzed_frames,
        frames_with_detections=frames_with_detections,
        detection_events=detection_events,
        average_confidence=round(confidence_sum / detection_events * 100, 1) if detection_events else 0.0,
        class_counts=class_counts,
        processing_ms=round(processing_ms, 1),
        processing_fps=round(frame_idx / (processing_ms / 1000), 2) if processing_ms > 0 else 0.0,
    )


def _run_video_job(job_id: str, input_path: str, playable_path: str, output_path: str) -> None:
    """Giới hạn một video xử lý tại một thời điểm để tránh quá tải CPU/RAM."""
    with _video_worker_semaphore:
        with _video_jobs_lock:
            if _video_jobs.get(job_id, {}).get("cancel_requested"):
                _video_jobs[job_id].update(
                    status="cancelled",
                    phase="cancelled",
                    error="Đã hủy do chọn video mới.",
                )
                return
        _process_video_job(job_id, input_path, playable_path, output_path)


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in VIDEO_ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="Định dạng video không được hỗ trợ.")

    # Chọn video mới thì hủy mọi job cũ ngay tại server, kể cả khi trình duyệt
    # không kịp gửi yêu cầu hủy trước đó.
    with _video_jobs_lock:
        for old_job in _video_jobs.values():
            if old_job["status"] == "processing":
                old_job["cancel_requested"] = True

    job_id        = uuid.uuid4().hex
    input_name    = f"{job_id}{ext}"
    playable_name = f"{job_id}_original.mp4"
    output_name   = f"{job_id}_annotated.mp4"
    input_path    = os.path.join(VIDEO_UPLOAD_DIR, input_name)
    playable_path = os.path.join(VIDEO_UPLOAD_DIR, playable_name)
    output_path   = os.path.join(VIDEO_OUTPUT_DIR, output_name)

    file_size = 0
    try:
        with open(input_path, "wb") as output_file:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > VIDEO_MAX_BYTES:
                    raise HTTPException(status_code=400, detail="File video vượt quá 100MB.")
                output_file.write(chunk)
    except Exception:
        if os.path.exists(input_path):
            os.remove(input_path)
        raise
    finally:
        await file.close()

    with _video_jobs_lock:
        _video_jobs[job_id] = {
            "status":       "processing",
            "phase":        "queued",
            "progress":     0.0,
            "error":        None,
            "cancel_requested": False,
            "filename":     file.filename or input_name,
            "file_size":    file_size,
            "processed_frames": 0,
            "original_url": f"/static/videos/uploads/{playable_name}",
            "result_url":   f"/static/videos/outputs/{output_name}",
        }

    threading.Thread(
        target=_run_video_job,
        args=(job_id, input_path, playable_path, output_path),
        daemon=True,
        name=f"ecovision-video-{job_id[:8]}",
    ).start()

    return JSONResponse({"job_id": job_id})


@app.post("/api/video-cancel/{job_id}")
async def cancel_video(job_id: str):
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy job xử lý video.")
        if job["status"] == "processing":
            job["cancel_requested"] = True
    return JSONResponse({"status": "ok"})


@app.get("/api/video-status/{job_id}")
async def video_status(job_id: str):
    with _video_jobs_lock:
        stored_job = _video_jobs.get(job_id)
        job = dict(stored_job) if stored_job is not None else None
    if job is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job xử lý video.")
    return JSONResponse(job)
