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
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# ─────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(BASE_DIR)
TRAIN_RUN_DIR = os.path.join(ROOT_DIR, "results", "figures", "PhanLoaiRac_5Class_v2")
MODEL_PATH    = os.path.join(TRAIN_RUN_DIR, "weights", "best.pt")
OPENVINO_DIR  = os.path.join(TRAIN_RUN_DIR, "weights", "best_openvino_model")
RESULTS_CSV   = os.path.join(TRAIN_RUN_DIR, "results.csv")

CAM_W         = 640
CAM_H         = 480
CAM_IMGSZ     = 640    # Khớp imgsz lúc train/test ảnh tĩnh → nhận diện chính xác hơn
CAM_CONF_DEF  = 0.35
CAM_MAX_DET   = 5
CAM_BOX_RATIO = 0.60   # Loại box chiếm > 60% diện tích frame

IMG_IMGSZ = 640
IMG_CONF  = 0.40

VIDEO_IMGSZ       = 640
VIDEO_CONF        = 0.40
VIDEO_MAX_DET     = 20
VIDEO_UPLOAD_DIR  = os.path.join(BASE_DIR, "static", "videos", "uploads")
VIDEO_OUTPUT_DIR  = os.path.join(BASE_DIR, "static", "videos", "outputs")
VIDEO_MAX_BYTES   = 100 * 1024 * 1024   # 100 MB
VIDEO_ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

_video_jobs_lock = threading.Lock()
_video_jobs: dict = {}   # job_id -> {status, progress, error, original_url, result_url}


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
def _load_model() -> YOLO:
    """Ưu tiên OpenVINO nếu có export, fallback sang PyTorch."""
    if os.path.isdir(OPENVINO_DIR):
        print(f"[EcoVision] OpenVINO model: {OPENVINO_DIR}")
        return YOLO(OPENVINO_DIR)
    print(f"[EcoVision] PyTorch model: {MODEL_PATH}")
    return YOLO(MODEL_PATH)

print("[EcoVision] Loading YOLO11n...")
model_cam   = _load_model()
model_img   = _load_model()       # instance riêng tránh shared predictor state
model_video = _load_model()       # instance riêng cho xử lý video upload
print("[EcoVision] Model ready.")

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


MERGE_MAP = {"CARDBOARD": "PAPER"}   # gộp carton vào giấy

def _draw_group_labels(frame: np.ndarray, result) -> np.ndarray:
    """Vẽ bounding box — CARDBOARD được hiển thị thành PAPER."""
    color_map = {
        "BIODEGRADABLE": (34,  197,  94),   # green
        "GLASS":         (168,  85, 247),   # purple
        "METAL":         (234, 179,   8),   # yellow
        "PAPER":         (249, 115,  22),   # orange
        "PLASTIC":       (239,  68,  68),   # red
    }

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        raw    = model_cam.names[cls_id]
        label  = MERGE_MAP.get(raw, raw)    # CARDBOARD → PAPER
        conf   = float(box.conf[0].item())
        display_text = f"{label} {conf * 100:.0f}%"
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_color = color_map.get(label, (150, 150, 150))

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
            result = model_cam.predict(
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
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ── Test-image endpoint (dev only) ────────────────────────────
_TEST_IMG_DIR = os.path.join(ROOT_DIR, "results", "test_images")
if os.path.isdir(_TEST_IMG_DIR):
    app.mount("/test-images", StaticFiles(directory=_TEST_IMG_DIR), name="test-images")

# ── Training artifacts (biểu đồ, ma trận nhầm lẫn...) ──────────
if os.path.isdir(TRAIN_RUN_DIR):
    app.mount("/train-artifacts", StaticFiles(directory=TRAIN_RUN_DIR), name="train-artifacts")

DATASET_CLASSES = ["BIODEGRADABLE", "GLASS", "METAL", "PAPER", "PLASTIC"]


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


@app.get("/api/cam-debug")
async def cam_debug():
    """Raw inference on latest frame — no _filter_boxes, returns box details."""
    with _raw_lock:
        frame = _latest_raw
    if frame is None:
        return JSONResponse({"error": "no frame"})
    fh, fw = frame.shape[:2]
    frame_area = fw * fh
    res = model_cam.predict(source=frame, conf=0.20, imgsz=CAM_IMGSZ,
                            max_det=20, agnostic_nms=True, verbose=False)[0]
    boxes = []
    for b in (res.boxes or []):
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        w, h = x2-x1, y2-y1
        ratio = (w*h) / frame_area
        boxes.append({
            "label": model_cam.names[int(b.cls[0])],
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

    return JSONResponse({
        "model":        "YOLO11n",
        "dataset":      "PhanLoaiRac_5Class_v2",
        "classes":      DATASET_CLASSES,
        "num_classes":  len(DATASET_CLASSES),
        "total_epochs": len(epochs),
        "best_epoch":   best,
        "last_epoch":   last,
        "history":      epochs,
        "dataset_counts": {
            "train": 6678,
            "valid": 1919,
            "test":  954,
        },
        "label_counts": {
            "BIODEGRADABLE": 7843,
            "GLASS":         7614,
            "PAPER":         6803,
            "PLASTIC":       5763,
            "METAL":         5736,
        },
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
    res = model_img.predict(
        source=pil_img,   # PIL RGB → Ultralytics tự chuyển nội bộ
        conf=IMG_CONF,
        imgsz=IMG_IMGSZ,
        max_det=300,
        verbose=False,
    )[0]
    total_ms = (time.perf_counter() - t0) * 1000

    # Dùng _draw_group_labels để bounding box label nhất quán với MERGE_MAP
    img_bgr       = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    annotated_bgr = _draw_group_labels(img_bgr, res)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    out_pil       = Image.fromarray(annotated_rgb)

    buf = io.BytesIO()
    out_pil.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    img_w, img_h = pil_img.size
    frame_area   = img_w * img_h

    detections: list[dict] = []
    if res.boxes and len(res.boxes) > 0:
        for box in res.boxes:
            cls_id   = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            label    = MERGE_MAP.get(model_img.names[cls_id], model_img.names[cls_id])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            box_area = (x2 - x1) * (y2 - y1)
            detections.append({
                "label":      label,
                "conf":       round(conf_val * 100, 1),
                "box":        [round(x1), round(y1), round(x2), round(y2)],
                "area_ratio": round(box_area / frame_area * 100, 1),
            })

    return JSONResponse({
        "image_b64":   img_b64,
        "detections":  detections,
        "count":       len(detections),
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
def _transcode_for_browser(input_path: str, playable_path: str) -> bool:
    """Chuyển video gốc (codec bất kỳ) sang H.264/MP4 để trình duyệt phát trực tiếp được."""
    r = subprocess.run(
        [
            FFMPEG_EXE, "-y", "-i", input_path,
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-movflags", "+faststart",
            playable_path,
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0 and os.path.exists(playable_path)


def _process_video_job(job_id: str, input_path: str, playable_path: str, output_path: str) -> None:
    def _set(**kwargs):
        with _video_jobs_lock:
            _video_jobs[job_id].update(kwargs)

    if not _transcode_for_browser(input_path, playable_path):
        _set(status="error", error="Không thể đọc hoặc chuyển đổi file video gốc.")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        _set(status="error", error="Không thể đọc file video.")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Encode qua ffmpeg (libx264) để trình duyệt phát được trực tiếp —
    # cv2.VideoWriter không có encoder H.264 khả dụng trên build OpenCV mặc định.
    ffmpeg_proc = subprocess.Popen(
        [
            FFMPEG_EXE, "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-movflags", "+faststart",
            output_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            result = model_video.predict(
                source=frame, conf=VIDEO_CONF, imgsz=VIDEO_IMGSZ,
                max_det=VIDEO_MAX_DET, agnostic_nms=True, verbose=False,
            )[0]
            frame_area = width * height
            _filter_boxes(result, frame_area)

            annotated = frame
            if result.boxes is not None and len(result.boxes) > 0:
                annotated = _draw_group_labels(frame, result)

            ffmpeg_proc.stdin.write(annotated.tobytes())

            if total_frames > 0:
                progress = round(frame_idx / total_frames * 100, 1)
                _set(progress=min(progress, 99.9))
    except Exception as exc:
        _set(status="error", error=f"Lỗi xử lý video: {exc}")
        return
    finally:
        cap.release()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

    if ffmpeg_proc.returncode != 0 or not os.path.exists(output_path):
        _set(status="error", error="Không thể tạo file video đầu ra.")
        return

    _set(status="done", progress=100.0)


@app.post("/api/upload-video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in VIDEO_ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="Định dạng video không được hỗ trợ.")

    data = await file.read()
    if len(data) > VIDEO_MAX_BYTES:
        raise HTTPException(status_code=400, detail="File video vượt quá 100MB.")

    job_id        = uuid.uuid4().hex
    input_name    = f"{job_id}{ext}"
    playable_name = f"{job_id}_original.mp4"
    output_name   = f"{job_id}_annotated.mp4"
    input_path    = os.path.join(VIDEO_UPLOAD_DIR, input_name)
    playable_path = os.path.join(VIDEO_UPLOAD_DIR, playable_name)
    output_path   = os.path.join(VIDEO_OUTPUT_DIR, output_name)

    with open(input_path, "wb") as f:
        f.write(data)

    with _video_jobs_lock:
        _video_jobs[job_id] = {
            "status":       "processing",
            "progress":     0.0,
            "error":        None,
            "original_url": f"/static/videos/uploads/{playable_name}",
            "result_url":   f"/static/videos/outputs/{output_name}",
        }

    background_tasks.add_task(_process_video_job, job_id, input_path, playable_path, output_path)

    return JSONResponse({"job_id": job_id})


@app.get("/api/video-status/{job_id}")
async def video_status(job_id: str):
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job xử lý video.")
    return JSONResponse(job)
