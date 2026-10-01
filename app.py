"""本地视觉训练平台 - 单文件后端

功能：
  - 图片 / 视频上传保存到 data/
  - YOLOv8 图片检测、视频检测（后台任务）
  - YOLOv8 模型训练（后台任务）
  - 静态页面（frontend/index.html）映射到 /

启动：python app.py
"""
import logging
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import vision

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("watch")

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VIDEO_DIR = vision.VIDEO_DIR
RESULTS_DIR = vision.RESULTS_DIR
OUTPUT_DIR = vision.OUTPUT_DIR
MODELS_DIR = vision.MODELS_DIR

IMAGE_EXTENSIONS = vision.IMAGE_EXTENSIONS
VIDEO_EXTENSIONS = vision.VIDEO_EXTENSIONS

app = FastAPI(title="本地视觉训练平台")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底异常处理：保证前端拿到结构化 JSON 错误，而不是空白页/HTML 错误页"""
    logger.exception("未处理异常：%s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误（{type(exc).__name__}）：{exc}"},
    )


def _unique_name(original: str, ext_set: set[str]) -> str:
    """校验扩展名并生成唯一文件名"""
    ext = Path(original or "").suffix.lower()
    if ext not in ext_set:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式 {ext or '(无扩展名)'}，仅支持：{', '.join(sorted(ext_set))}",
        )
    return f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"


def _save_upload(file: UploadFile, directory: Path, ext_set: set[str]) -> str:
    filename = _unique_name(file.filename or "", ext_set)
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / filename
    try:
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"保存文件失败: {e}")
    return filename


def _safe_file(directory: Path, filename: str) -> Path:
    """防目录穿越地取文件"""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    path = directory / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return path


def _parse_classes(classes: str | None) -> list[str] | None:
    """把 "person, car, 书本" 解析成类别列表（中英文逗号都支持），空则返回 None"""
    if not classes:
        return None
    items = [c.strip() for c in classes.replace("，", ",").split(",")]
    items = [c for c in items if c]
    return items or None


def _clamp_conf(conf: float) -> float:
    """置信度钳制到 (0, 1)：ultralytics 对越界值会直接抛 ValueError（如 conf=111）"""
    return max(0.01, min(conf, 0.99))


# ---------- YOLO 状态 ----------
@app.get("/api/yolo/status", summary="YOLO 环境状态")
def yolo_status() -> dict:
    available, message = vision.yolo_available()
    models: list[str] = []
    world_models: list[str] = []
    if MODELS_DIR.is_dir():
        for p in sorted(MODELS_DIR.glob("*.pt")):
            models.append(p.name)
            if vision.is_world_weights(p.name):
                world_models.append(p.name)
    return {
        "available": available,
        "message": message,
        "models": models,
        "world_models": world_models,           # 属于 YOLO-World 的权重
        "default_model": vision.DEFAULT_WEIGHTS,  # 前端下拉框默认选中它
        "clip_available": vision.clip_available(),  # World 所需的文本编码器依赖
    }


# ---------- 上传 ----------
@app.post("/api/upload", summary="上传图片")
async def upload_image(file: UploadFile = File(...)) -> dict:
    filename = _save_upload(file, UPLOAD_DIR, IMAGE_EXTENSIONS)
    return {"filename": filename, "url": f"/images/{filename}"}


@app.post("/api/upload_video", summary="上传视频")
async def upload_video(file: UploadFile = File(...)) -> dict:
    filename = _save_upload(file, VIDEO_DIR, VIDEO_EXTENSIONS)
    return {"filename": filename, "url": f"/videos/{filename}"}


# ---------- 列表 ----------
@app.get("/api/images", summary="图片 + 结果列表")
def list_images() -> dict:
    files = (
        sorted((p.name for p in UPLOAD_DIR.iterdir() if p.is_file()), reverse=True)
        if UPLOAD_DIR.is_dir()
        else []
    )
    results = (
        sorted((p.name for p in RESULTS_DIR.iterdir() if p.is_file()), reverse=True)
        if RESULTS_DIR.is_dir()
        else []
    )
    return {
        "count": len(files),
        "images": [{"filename": f, "url": f"/images/{f}"} for f in files],
        "results": [{"filename": f, "url": f"/results/{f}"} for f in results],
    }


@app.get("/api/videos", summary="视频 + 输出列表")
def list_videos() -> dict:
    files = (
        sorted((p.name for p in VIDEO_DIR.iterdir() if p.is_file()), reverse=True)
        if VIDEO_DIR.is_dir()
        else []
    )
    outputs = (
        sorted((p.name for p in OUTPUT_DIR.iterdir() if p.is_file()), reverse=True)
        if OUTPUT_DIR.is_dir()
        else []
    )
    return {
        "count": len(files),
        "videos": [{"filename": f, "url": f"/videos/{f}"} for f in files],
        "outputs": [{"filename": f, "url": f"/outputs/{f}"} for f in outputs],
    }


# ---------- 文件访问 ----------
@app.get("/images/{filename}", summary="访问图片")
def get_image(filename: str) -> FileResponse:
    return FileResponse(_safe_file(UPLOAD_DIR, filename))


@app.get("/results/{filename}", summary="访问检测结果图")
def get_result(filename: str) -> FileResponse:
    return FileResponse(_safe_file(RESULTS_DIR, filename))


@app.get("/videos/{filename}", summary="访问视频")
def get_video(filename: str) -> FileResponse:
    return FileResponse(_safe_file(VIDEO_DIR, filename))


@app.get("/outputs/{filename}", summary="访问检测结果视频")
def get_output(filename: str) -> FileResponse:
    return FileResponse(_safe_file(OUTPUT_DIR, filename))


@app.get("/models/{run}/{filename}", summary="下载训练得到的权重")
def get_model_file(run: str, filename: str) -> FileResponse:
    if "/" in run or "\\" in run or ".." in run:
        raise HTTPException(status_code=400, detail="非法路径")
    return FileResponse(_safe_file(vision.RUNS_DIR / run / "weights", filename))


# ---------- 检测 ----------
@app.post("/api/detect/image/{filename}", summary="检测图片")
def detect_image(
    filename: str,
    conf: float = 0.25,
    weights: str = vision.DEFAULT_WEIGHTS,
    classes: str | None = None,
) -> dict:
    available, message = vision.yolo_available()
    if not available:
        raise HTTPException(status_code=503, detail=f"YOLO 不可用：{message}")

    ok, reason = vision.world_ready(weights)
    if not ok:
        raise HTTPException(status_code=503, detail=reason)

    conf = _clamp_conf(conf)
    class_list = _parse_classes(classes)

    src = _safe_file(UPLOAD_DIR, filename)
    logger.info(
        "开始检测图片：%s（conf=%s, weights=%s, classes=%s）",
        filename,
        conf,
        weights,
        class_list,
    )
    t0 = time.perf_counter()

    try:
        result = vision.detect_image(src, weights=weights, conf=conf, classes=class_list)
    except Exception as e:  # noqa: BLE001
        logger.exception("检测图片失败：%s", filename)
        raise HTTPException(
            status_code=500, detail=f"检测失败（{type(e).__name__}）：{e}"
        ) from e

    elapsed = time.perf_counter() - t0
    logger.info(
        "检测完成：%s 模式=%s 检出 %s 个目标，耗时 %.2fs",
        filename,
        result.get("mode"),
        result.get("count"),
        elapsed,
    )
    result["elapsed_seconds"] = round(elapsed, 3)
    return result


@app.post("/api/detect/video/{filename}", summary="检测视频（后台任务）")
def detect_video(
    filename: str,
    conf: float = 0.25,
    stride: int = 1,
    weights: str = vision.DEFAULT_WEIGHTS,
    classes: str | None = None,
) -> dict:
    available, message = vision.yolo_available()
    if not available:
        raise HTTPException(status_code=503, detail=f"YOLO 不可用：{message}")

    ok, reason = vision.world_ready(weights)
    if not ok:
        raise HTTPException(status_code=503, detail=reason)

    src = _safe_file(VIDEO_DIR, filename)
    stride = max(1, min(stride, 30))
    conf = _clamp_conf(conf)
    class_list = _parse_classes(classes)

    try:
        job_id = vision.start_video_job(
            src, weights=weights, conf=conf, stride=stride, classes=class_list
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("提交视频检测任务失败：%s", filename)
        raise HTTPException(
            status_code=500, detail=f"提交失败（{type(e).__name__}）：{e}"
        ) from e

    logger.info(
        "已提交视频检测任务：%s（job=%s, weights=%s, conf=%s, stride=%s, classes=%s）",
        filename,
        job_id,
        weights,
        conf,
        stride,
        class_list,
    )
    return {"job_id": job_id}


# ---------- 训练 ----------
@app.get("/api/datasets", summary="数据集列表")
def list_datasets() -> dict:
    datasets = vision.list_datasets()
    return {"count": len(datasets), "datasets": datasets}


@app.post("/api/train", summary="启动训练（后台任务）")
def start_train(
    dataset: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    lr0: float = 0.01,
    weights: str = vision.DEFAULT_WEIGHTS,
) -> dict:
    available, message = vision.yolo_available()
    if not available:
        raise HTTPException(status_code=503, detail=f"YOLO 不可用：{message}")
    if not (vision.DATASETS_DIR / dataset).is_dir():
        raise HTTPException(status_code=404, detail=f"数据集不存在：{dataset}")
    if epochs < 1 or epochs > 1000:
        raise HTTPException(status_code=400, detail="epochs 需在 1-1000 之间")
    job_id = vision.start_train_job(
        dataset=dataset, weights=weights, epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr0
    )
    return {"job_id": job_id}


# ---------- 任务状态 ----------
@app.get("/api/jobs", summary="任务列表")
def list_jobs() -> dict:
    jobs = sorted(vision.JOBS.values(), key=lambda j: j["id"], reverse=True)
    return {"count": len(jobs), "jobs": jobs}


@app.get("/api/jobs/{job_id}", summary="任务状态")
def get_job(job_id: str) -> dict:
    job = vision.JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


# ---------- 静态页面（http://127.0.0.1:8000）----------
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


# ---------- 启动 ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
