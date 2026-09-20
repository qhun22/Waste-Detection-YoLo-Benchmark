"""
EcoVision AI — FastAPI Backend
Phân loại rác thải thời gian thực bằng YOLO trên CPU.

Cài đặt:
    pip install fastapi uvicorn[standard] python-multipart jinja2 opencv-python pillow ultralytics

Khởi động:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import base64
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

# ─────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_CAM_PATH = os.path.join(BASE_DIR, "results", "weights", "yolo11n_best.pt")
MODEL_IMG_PATH = os.path.join(BASE_DIR, "results", "weights", "yolov8s_best.pt")
OPENVINO_DIR   = os.path.join(BASE_DIR, "results", "weights", "yolo11n_best_openvino_model")

CAM_W         = 640
CAM_H         = 480
CAM_IMGSZ     = 256    # Kích thước nhỏ hơn để tăng FPS trên CPU
CAM_CONF_DEF  = 0.45
CAM_MAX_DET   = 5
CAM_BOX_RATIO = 0.60   # Loại box chiếm > 60% diện tích frame

IMG_IMGSZ = 640
IMG_CONF  = 0.35

CATEGORY_MAPPING: dict[str, tuple[str, str]] = {
    "BIODEGRADABLE": ("Rác Hữu Cơ",         "🟢 Thùng Xanh Lá"),
    "CARDBOARD":     ("Rác Tái Chế / Vô Cơ", "🔵 Thùng Xanh Dương"),
    "GLASS":         ("Rác Tái Chế / Vô Cơ", "🔵 Thùng Xanh Dương"),
    "METAL":         ("Rác Tái Chế / Vô Cơ", "🔵 Thùng Xanh Dương"),
    "PAPER":         ("Rác Tái Chế / Vô Cơ", "🔵 Thùng Xanh Dương"),
    "PLASTIC":       ("Rác Tái Chế / Vô Cơ", "🔵 Thùng Xanh Dương"),
}

# ─────────────────────────────────────────────────────────────────────
#  Thread-safe Shared State
# ─────────────────────────────────────────────────────────────────────
_raw_lock   = threading.Lock()
_latest_raw: np.ndarray | None = None   # BGR frame từ OpenCV

_ann_lock   = threading.Lock()
_latest_ann: np.ndarray | None = None   # BGR frame đã annotate

_stats_lock = threading.Lock()
_stats: dict = {"fps": 0.0, "detections": 0, "conf": CAM_CONF_DEF}

# ─────────────────────────────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────────────────────────────
def _load_cam_model() -> YOLO:
    """Ưu tiên OpenVINO nếu có export, fallback sang PyTorch."""
    if os.path.isdir(OPENVINO_DIR):
        print(f"[EcoVision] OpenVINO model: {OPENVINO_DIR}")
        return YOLO(OPENVINO_DIR)
    print(f"[EcoVision] PyTorch model: {MODEL_CAM_PATH}")
    return YOLO(MODEL_CAM_PATH)

print("[EcoVision] Loading models...")
model_cam = _load_cam_model()
model_img  = YOLO(MODEL_IMG_PATH)
print("[EcoVision] Both models ready.")

# ─────────────────────────────────────────────────────────────────────
#  Helper: Lọc bounding box bất thường
# ─────────────────────────────────────────────────────────────────────
def _filter_boxes(result, frame_area: int) -> None:
    """
    Lọc in-place:
      1. Box diện tích > CAM_BOX_RATIO → background / toàn thân người
      2. Aspect ratio cực đoan → tay/chân người dọc theo cạnh frame
    """
    if result.boxes is None or len(result.boxes) == 0:
        return
    xyxy   = result.boxes.xyxy
    w      = xyxy[:, 2] - xyxy[:, 0]
    h      = xyxy[:, 3] - xyxy[:, 1]
    areas  = w * h
    aspect = w / (h + 1e-6)
    keep   = (
        (areas / frame_area <= CAM_BOX_RATIO) &
        (aspect >= 0.12) &
        (aspect <= 7.0)
    )
    result.boxes = Boxes(result.boxes.data[keep], result.boxes.orig_shape)


def _draw_group_labels(frame: np.ndarray, result) -> np.ndarray:
    """Vẽ bounding box và tên nhóm rác bằng tiếng Việt không dấu."""
    group_map = {
        "BIODEGRADABLE": "Rac Huu Co",
        "PLASTIC": "Rac Vo Co / Tai Che",
        "PAPER": "Rac Vo Co / Tai Che",
        "METAL": "Rac Vo Co / Tai Che",
        "GLASS": "Rac Vo Co / Tai Che",
        "CARDBOARD": "Rac Vo Co / Tai Che",
    }
    color_map = {
        "Rac Huu Co": (0, 200, 0),
        "Rac Vo Co / Tai Che": (255, 140, 0),
        "Rac Khac": (0, 0, 255),
    }

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        label = model_cam.names[cls_id]
        conf = float(box.conf[0].item())
        group = group_map.get(label, "Rac Khac")
        display_text = f"{group} ({label} {conf * 100:.0f}%)"
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_color = color_map[group]

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
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print("[EcoVision] ERROR: Cannot open webcam (index 0)!")
        return

    print("[EcoVision] Camera thread started.")
    while True:
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
            res = model_cam.predict(
                source=frame,      # BGR numpy array — YOLO xử lý natively
                conf=conf,
                imgsz=CAM_IMGSZ,
                max_det=CAM_MAX_DET,
                agnostic_nms=True,
                verbose=False,
            )[0]

            _filter_boxes(res, frame_area)
            n_det     = len(res.boxes) if res.boxes else 0
            annotated = _draw_group_labels(frame.copy(), res)

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

# ─────────────────────────────────────────────────────────────────────
#  MJPEG Stream Generator
# ─────────────────────────────────────────────────────────────────────
def _make_placeholder() -> np.ndarray:
    img  = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    text = "Dang khoi dong camera..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, 0.8, 2)
    cv2.putText(img, text, ((CAM_W - tw) // 2, CAM_H // 2),
                font, 0.8, (150, 150, 150), 2)
    return img

_PLACEHOLDER = _make_placeholder()

def _mjpeg_gen():
    """
    Yields MJPEG boundary-separated JPEG frames ở tối đa 30 FPS.
    Màu sắc: BGR annotated → cv2.imencode → browser hiển thị chuẩn RGB.
    (OpenCV JPEG encoder tự xử lý BGR→display conversion.)
    """
    while True:
        with _ann_lock:
            frame = _latest_ann

        src = frame if frame is not None else _PLACEHOLDER
        ok, buf = cv2.imencode(".jpg", src, [cv2.IMWRITE_JPEG_QUALITY, 82])
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
app.mount("/static",  StaticFiles(directory="static"),    name="static")
templates = Jinja2Templates(directory="templates")


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
        return JSONResponse(dict(_stats))


@app.post("/api/set-conf")
async def set_conf(conf: float = CAM_CONF_DEF):
    val = round(max(0.10, min(0.90, conf)), 2)
    with _stats_lock:
        _stats["conf"] = val
    return JSONResponse({"status": "ok", "conf": val})


@app.post("/api/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        data    = await file.read()
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File ảnh không hợp lệ.")

    res = model_img.predict(
        source=pil_img,   # PIL RGB → Ultralytics tự chuyển nội bộ
        conf=IMG_CONF,
        imgsz=IMG_IMGSZ,
        verbose=False,
    )[0]

    # result.plot() → BGR → chuyển RGB để PIL/base64 hiển thị đúng màu
    annotated_bgr = res.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    out_pil       = Image.fromarray(annotated_rgb)

    buf = io.BytesIO()
    out_pil.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    detections: list[dict] = []
    if res.boxes and len(res.boxes) > 0:
        for box in res.boxes:
            cls_id   = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            label    = model_img.names[cls_id]
            cat, bin_ = CATEGORY_MAPPING.get(label, ("Rác Khác", "🟠 Thùng Cam"))
            detections.append({
                "label":    label,
                "conf":     round(conf_val * 100, 1),
                "category": cat,
                "bin":      bin_,
            })

    return JSONResponse({
        "image_b64":  img_b64,
        "detections": detections,
        "count":      len(detections),
    })
