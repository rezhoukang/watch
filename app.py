"""本地视觉训练平台 - 单文件后端

功能：
  - 图片 / 视频上传保存到 data/
  - YOLOv8 图片推理、视频推理（后台任务）、RT-DETRv4 图片人头计数
  - YOLOv8 模型训练（后台任务）
  - 静态页面（frontend/index.html）映射到 /

启动：python app.py
"""
import logging
import re
import shutil
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import vision
import rtdetrv4
import inference_history

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


# Windows 文件名非法字符；同时挡掉路径分隔符，防目录穿越
_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Windows 保留设备名，不能直接当文件名
_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _safe_name(original: str, ext_set: set[str]) -> str:
    """沿用上传时的原文件名（只做安全清洗），扩展名必须在白名单内"""
    name = Path(original or "").name.strip()      # 去掉任何路径部分
    suffix = Path(name).suffix
    if suffix.lower() not in ext_set:
        raise HTTPException(
            status_code=400,
            detail=(
                f"不支持的文件格式 {suffix or '(无扩展名)'}，"
                f"仅支持：{', '.join(sorted(ext_set))}"
            ),
        )
    stem = _ILLEGAL_CHARS.sub("_", Path(name).stem).strip(" .")  # 结尾不能是点/空格
    if not stem:
        raise HTTPException(status_code=400, detail=f"文件名不合法：{original}")
    if stem.upper() in _RESERVED_NAMES:
        stem = f"_{stem}"
    return f"{stem}{suffix}"


def _save_upload(file: UploadFile, directory: Path, ext_set: set[str]) -> tuple[str, bool]:
    """按原文件名保存（同名直接覆盖）。返回 (文件名, 是否覆盖了已有文件)"""
    filename = _safe_name(file.filename or "", ext_set)
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / filename
    overwritten = dest.exists()
    try:
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"保存文件失败: {e}")
    return filename, overwritten


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


def _require_image_backend(weights: str) -> None:
    if rtdetrv4.is_weights(weights):
        state = rtdetrv4.status()
        if not state["available"]:
            raise HTTPException(status_code=503, detail=state["message"])
        return
    available, message = vision.yolo_available()
    if not available:
        raise HTTPException(status_code=503, detail=f"YOLO 不可用：{message}")
    ok, reason = vision.world_ready(weights)
    if not ok:
        raise HTTPException(status_code=503, detail=reason)


def _require_yolo_weights(weights: str) -> None:
    if rtdetrv4.is_weights(weights):
        raise HTTPException(status_code=400, detail="RT-DETRv4 当前仅支持图片人数统计，请在图片推理页使用")


# ---------- YOLO 状态 ----------
@app.get("/api/yolo/status", summary="YOLO 环境状态")
def yolo_status() -> dict:
    available, message = vision.yolo_available()
    weights: list[str] = []
    world_weights: list[str] = []
    if MODELS_DIR.is_dir():
        for p in sorted(MODELS_DIR.glob("*.pt")):
            weights.append(p.name)
            if vision.is_world_weights(p.name):
                world_weights.append(p.name)
    head_model = rtdetrv4.status()
    return {
        "available": available,
        "message": message,
        "weights": weights,
        "world_weights": world_weights,             # 属于 YOLO-World 的权重
        "default_weights": vision.DEFAULT_WEIGHTS,  # 前端下拉框默认选中它
        "clip_available": vision.clip_available(),  # World 所需的文本编码器依赖
        "image_weights": weights + [rtdetrv4.WEIGHTS_NAME],
        "default_image_weights": rtdetrv4.WEIGHTS_NAME if head_model["available"] else vision.DEFAULT_WEIGHTS,
        "model_labels": {rtdetrv4.WEIGHTS_NAME: rtdetrv4.MODEL_NAME},
        "rtdetrv4": head_model,
    }


@app.get("/api/rtdetrv4/status", summary="RT-DETRv4 人头检测环境状态")
def rtdetrv4_status() -> dict:
    return rtdetrv4.status()


# ---------- 上传 ----------
@app.post("/api/upload", summary="上传图片")
async def upload_image(file: UploadFile = File(...)) -> dict:
    filename, overwritten = _save_upload(file, UPLOAD_DIR, IMAGE_EXTENSIONS)
    logger.info("上传图片：%s%s", filename, "（覆盖同名文件）" if overwritten else "")
    return {"filename": filename, "url": f"/images/{filename}", "overwritten": overwritten}


@app.post("/api/upload_video", summary="上传视频")
async def upload_video(file: UploadFile = File(...)) -> dict:
    filename, overwritten = _save_upload(file, VIDEO_DIR, VIDEO_EXTENSIONS)
    logger.info("上传视频：%s%s", filename, "（覆盖同名文件）" if overwritten else "")
    return {"filename": filename, "url": f"/videos/{filename}", "overwritten": overwritten}


@app.delete("/api/images/{filename}", summary="删除图片")
def delete_image(filename: str) -> dict:
    """删除待测图片（已生成的推理结果图不动）"""
    path = _safe_file(UPLOAD_DIR, filename)
    try:
        path.unlink()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"删除失败：{e}") from e
    logger.info("已删除图片：%s", filename)
    return {"ok": True, "filename": filename}


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
    # 扫描 uploads 里视频检测提取出的「最佳帧」（<视频名>_best_<帧号>.jpg）
    best_frames: dict[str, str] = {}
    if UPLOAD_DIR.is_dir():
        for p in UPLOAD_DIR.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and "_best_" in p.name:
                orig = p.name.split("_best_", 1)[0]
                best_frames.setdefault(orig, p.name)
    out_list: list[dict] = []
    frames_dir = vision.OUTPUT_DIR / "frames"
    for f in outputs:
        entry: dict = {"filename": f, "url": f"/outputs/{f}"}
        # 人数折线图数据：<视频输出名>.json 侧车（每帧 head 计数）
        chart_json = frames_dir / f"{Path(f).stem}.json"
        if chart_json.is_file():
            entry["chart_url"] = f"/outputs/frames/{chart_json.name}"
        # 「图预览」：输出视频名去掉 _det... 后即原始视频名，匹配它提取的最佳帧
        orig = f.split("_det", 1)[0]
        bf = best_frames.get(orig)
        if bf:
            entry["best_frame"] = bf
            entry["best_frame_url"] = f"/images/{bf}"
        out_list.append(entry)
    return {
        "count": len(files),
        "videos": [{"filename": f, "url": f"/videos/{f}"} for f in files],
        "outputs": out_list,
    }


def _history_record(record_id: str) -> dict:
    try:
        return inference_history.read_record(record_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="推理记录不存在") from e


@app.get("/api/inferences", summary="历史推理结果图片列表")
def list_inferences() -> dict:
    records = inference_history.list_records()
    return {"count": len(records), "records": records}


@app.post("/api/inferences/merge", summary="合并两次推理（重叠保留基准框）")
def merge_inferences(payload: dict) -> dict:
    base_id, incoming_id = payload.get("base_id"), payload.get("incoming_id")
    if not isinstance(base_id, str) or not isinstance(incoming_id, str):
        raise HTTPException(status_code=400, detail="请选择基准和待合并记录")
    try:
        return inference_history.merge_records(base_id, incoming_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="推理记录不存在") from e


@app.get("/api/inferences/{record_id}", summary="查看历史推理详情")
def get_inference(record_id: str) -> dict:
    return _history_record(record_id)


@app.get("/api/inferences/{record_id}/image/{kind}", summary="访问历史推理原图或结果图")
def get_inference_image(record_id: str, kind: str) -> FileResponse:
    record = _history_record(record_id)
    if kind == "source":
        path = inference_history.source_path(record)
    elif kind == "result":
        path = inference_history.record_dir(record_id) / "result.png"
    else:
        raise HTTPException(status_code=400, detail="图片类型只能为 source 或 result")
    return FileResponse(path)


@app.get("/api/videos/{filename}/info", summary="视频元信息（时长/分辨率/帧率）")
def video_info(filename: str) -> dict:
    """给前端时间轴用的：总时长（秒）、宽高、帧率、总帧数"""
    path = _safe_file(VIDEO_DIR, filename)
    info = vision.video_info(path)
    if info is None:
        raise HTTPException(status_code=400, detail="无法读取视频信息（文件损坏或格式不支持）")
    return info


@app.delete("/api/videos/{filename}", summary="删除视频")
def delete_video(filename: str) -> dict:
    """删除上传的视频（不删已生成的推理输出）"""
    path = _safe_file(VIDEO_DIR, filename)
    try:
        path.unlink()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"删除失败：{e}") from e
    logger.info("已删除视频：%s", filename)
    return {"ok": True, "filename": filename}


# ---------- 文件访问 ----------
@app.get("/images/{filename}", summary="访问图片")
def get_image(filename: str) -> FileResponse:
    return FileResponse(_safe_file(UPLOAD_DIR, filename))


@app.get("/results/{filename}", summary="访问推理结果图")
def get_result(filename: str) -> FileResponse:
    return FileResponse(_safe_file(RESULTS_DIR, filename))


@app.get("/videos/{filename}", summary="访问视频")
def get_video(filename: str) -> FileResponse:
    return FileResponse(_safe_file(VIDEO_DIR, filename))


@app.get("/outputs/frames/{filename}", summary="访问视频折线图数据（每帧人数）")
def get_output_frame_data(filename: str) -> FileResponse:
    """访问 data/outputs/frames/ 下的折线图 json（head_counts）"""
    return FileResponse(_safe_file(vision.OUTPUT_DIR / "frames", filename))


@app.get("/outputs/{filename}", summary="访问推理结果视频")
def get_output(filename: str) -> FileResponse:
    return FileResponse(_safe_file(OUTPUT_DIR, filename))


@app.get("/models/{run}/{filename}", summary="下载训练得到的权重")
def get_model_file(run: str, filename: str) -> FileResponse:
    if "/" in run or "\\" in run or ".." in run:
        raise HTTPException(status_code=400, detail="非法路径")
    return FileResponse(_safe_file(vision.RUNS_DIR / run / "weights", filename))


# ---------- 推理 ----------
@app.post("/api/detect/image/{filename}", summary="推理图片")
def detect_image(
    filename: str,
    conf: float = 0.38,
    weights: str = vision.DEFAULT_WEIGHTS,
    classes: str | None = None,
    imgsz: int = 640,
) -> dict:
    _require_image_backend(weights)

    conf = _clamp_conf(conf)
    class_list = _parse_classes(classes)

    src = _safe_file(UPLOAD_DIR, filename)
    logger.info(
        "开始推理图片：%s（conf=%s, weights=%s, imgsz=%s, classes=%s）",
        filename,
        conf,
        weights,
        imgsz,
        class_list,
    )
    t0 = time.perf_counter()

    try:
        record_id, snapshot = inference_history.begin_record(src)
        result = vision.detect_image(
            snapshot, weights=weights, conf=conf, classes=class_list, imgsz=imgsz, save_result=False
        )
        result["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        result = inference_history.complete_record(
            record_id, snapshot, result,
            {"confidence": conf, "weights": weights, "classes": class_list, "imgsz": imgsz},
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("推理图片失败：%s", filename)
        raise HTTPException(
            status_code=500, detail=f"推理失败（{type(e).__name__}）：{e}"
        ) from e

    elapsed = time.perf_counter() - t0
    logger.info(
        "推理完成：%s 模式=%s 检出 %s 个目标，耗时 %.2fs",
        filename,
        result.get("mode"),
        result.get("count"),
        elapsed,
    )
    return result


@app.post("/api/detect/video/{filename}", summary="推理视频（后台任务）")
def detect_video(
    filename: str,
    conf: float = 0.38,
    stride: int = 1,
    weights: str = vision.DEFAULT_WEIGHTS,
    classes: str | None = None,
    start: float = 0.0,
    end: float | None = None,
    render: str = "box",
    line_width: int = 2,
) -> dict:
    _require_yolo_weights(weights)
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

    # 显示形式：'box' 框 / 'dot' 中心点；line_width 钳到 1~8
    if render not in ("box", "dot"):
        raise HTTPException(status_code=400, detail="render 只能是 box 或 dot")
    line_width = max(1, min(line_width, 8))

    # 时间范围校验：end 缺省 = 全片；必须 start < end
    start = max(0.0, start)
    if end is not None and end <= start:
        raise HTTPException(status_code=400, detail="时间范围无效：end 必须大于 start")

    try:
        job_id = vision.start_video_job(
            src,
            weights=weights,
            conf=conf,
            stride=stride,
            classes=class_list,
            start=start,
            end=end,
            render=render,
            line_width=line_width,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("提交视频推理任务失败：%s", filename)
        raise HTTPException(
            status_code=500, detail=f"提交失败（{type(e).__name__}）：{e}"
        ) from e

    logger.info(
        "已提交视频推理任务：%s（job=%s, weights=%s, conf=%s, stride=%s, classes=%s, "
        "range=%s~%ss, render=%s, lw=%s）",
        filename,
        job_id,
        weights,
        conf,
        stride,
        class_list,
        start,
        end,
        render,
        line_width,
    )
    return {"job_id": job_id}


# ---------- 训练 ----------
@app.get("/api/datasets", summary="数据集列表")
def list_datasets() -> dict:
    datasets = vision.list_datasets()
    return {"count": len(datasets), "datasets": datasets}


@app.get("/api/datasets/{name}", summary="数据集详情（含 Train/Val 子集与图片列表）")
def dataset_detail(name: str) -> dict:
    ds_dir = vision.DATASETS_DIR / name
    if not ds_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"数据集不存在：{name}")
    return vision.dataset_detail(ds_dir)


@app.put("/api/datasets/{name}/classes", summary="改写数据集类别名")
def write_dataset_classes(name: str, payload: dict) -> dict:
    """body: {"classes": ["person", "flag", ...]}

    只改名字，class_id（行号）不变，所以已有 label 文件不用重写；
    后端会一并同步 data.yaml 的 nc/names 并删掉训练标注缓存。
    """
    ds_dir = vision.DATASETS_DIR / name
    if not ds_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"数据集不存在：{name}")
    names = payload.get("classes")
    if not isinstance(names, list):
        raise HTTPException(status_code=400, detail="classes 必须是数组")
    try:
        saved = vision.write_dataset_classes(ds_dir, names)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    logger.info("数据集 %s 的类别已更新：%s", name, saved)
    return {"ok": True, "classes": saved}


# ---------- 标注（训练数据准备）----------
# 注意：filename 形如 <数据集>/<split>/<图片名>，含斜杠，所以用 {filename:path}
# 与推理共用同一套权重选项（conf/weights/classes/imgsz）。
@app.get("/api/annotate/image/{filename:path}", summary="标注器：取一张图片的现有标注")
def annotate_image(filename: str) -> dict:
    """filename 形如 <dataset>/<split>/<img>，如 demo/train/abc.jpg
    返回该图片已有标注（YOLO 坐标已还原为原图像素）+ 类别清单。
    """
    img_path = vision.resolve_annotate_image(filename)
    labels = vision.read_labels_for_image(img_path)
    ds_dir = vision.find_dataset_dir(img_path)
    classes = vision.dataset_classes(ds_dir)
    return {
        "ok": True,
        "dataset": ds_dir.name,
        "split": vision.split_of(img_path),
        "filename": img_path.name,
        "url": f"/api/annotate/raw/{filename}",
        "width": labels["width"],
        "height": labels["height"],
        "detections": labels["detections"],
        "classes": classes,
    }


@app.get("/api/annotate/raw/{filename:path}", summary="访问标注器里的原图")
def get_annotate_image(filename: str):
    return FileResponse(vision.resolve_annotate_image(filename))


@app.post("/api/annotate/prelabel/{filename:path}", summary="标注器：对当前图片做预标注")
def prelabel_image(
    filename: str,
    conf: float = 0.38,
    weights: str = vision.DEFAULT_WEIGHTS,
    classes: str | None = None,
    imgsz: int = 640,
) -> dict:
    """用与推理一致的权重选项对单张图片做推理，结果转成标注框返回。
    filename 形如 <dataset>/<split>/<img>。
    """
    _require_yolo_weights(weights)
    img_path = vision.resolve_annotate_image(filename)
    available, message = vision.yolo_available()
    if not available:
        raise HTTPException(status_code=503, detail=f"YOLO 不可用：{message}")
    ok, reason = vision.world_ready(weights)
    if not ok:
        raise HTTPException(status_code=503, detail=reason)

    conf = _clamp_conf(conf)
    class_list = _parse_classes(classes)
    ds_dir = vision.find_dataset_dir(img_path)
    ds_classes = vision.dataset_classes(ds_dir)

    try:
        dets = vision.prelabel_image(
            img_path, weights=weights, conf=conf, classes=class_list, imgsz=imgsz
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("预标注失败：%s", filename)
        raise HTTPException(
            status_code=500, detail=f"预标注失败（{type(e).__name__}）：{e}"
        ) from e

    # 模型输出 → 数据集类别：命中就用；未命中则追加为新类别（不重排、不影响已有 id）
    unknown = [d["class_name"] for d in dets if d["class_name"] not in ds_classes]
    if unknown:
        ds_classes = vision.append_dataset_classes(ds_dir, unknown)
        logger.info("预标注新增类别：%s", sorted(set(unknown)))
    ds_name_to_id = {n: i for i, n in enumerate(ds_classes)}
    mapped = []
    for d in dets:
        cid = ds_name_to_id.get(d["class_name"])
        if cid is None:
            cid = 0
            d["class_name"] = ds_classes[0] if ds_classes else str(0)
        d["class_id"] = cid
        mapped.append(d)

    logger.info(
        "预标注完成：%s（conf=%s, weights=%s, 检出 %s 个）",
        filename,
        conf,
        weights,
        len(mapped),
    )
    return {"ok": True, "detections": mapped, "classes": ds_classes}


@app.post("/api/annotate/save/{filename:path}", summary="标注器：保存标注到 YOLO label 文件")
def annotate_save(filename: str, payload: dict) -> dict:
    """body: {"detections": [{"class_id", "bbox": [x1,y1,x2,y2]}, ...]}
    保存为 <数据集>/labels/<split>/<图名>.txt（YOLO 归一化格式）。
    """
    img_path = vision.resolve_annotate_image(filename)
    dets = payload.get("detections", [])
    if not isinstance(dets, list):
        raise HTTPException(status_code=400, detail="detections 必须是数组")
    saved, lbl_rel = vision.save_labels_for_image(img_path, dets)
    logger.info("保存标注：%s → %s（%s 个目标）", filename, lbl_rel, saved)
    return {"ok": True, "count": saved, "label": lbl_rel}


@app.post("/api/train", summary="启动训练（后台任务）")
def start_train(
    dataset: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    lr0: float = 0.01,
    weights: str = vision.DEFAULT_WEIGHTS,
) -> dict:
    _require_yolo_weights(weights)
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


@app.post("/api/jobs/{job_id}/cancel", summary="取消任务")
def cancel_job(job_id: str) -> dict:
    """取消运行中的视频推理 / 训练任务（后台线程会在下一帧/下一轮停下）"""
    if job_id not in vision.JOBS:
        raise HTTPException(status_code=404, detail="任务不存在")
    ok = vision.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="任务已结束，无法取消")
    logger.info("已请求取消任务：%s", job_id)
    return {"ok": True, "job_id": job_id}


@app.delete("/api/outputs/{filename}", summary="删除推理结果视频")
def delete_output(filename: str) -> dict:
    """删除 data/outputs/ 里的推理结果视频（连同折线图数据侧车 json）"""
    path = _safe_file(OUTPUT_DIR, filename)
    try:
        path.unlink()
        # 折线图数据侧车（<视频stem>.json）
        side = vision.OUTPUT_DIR / "frames" / f"{Path(filename).stem}.json"
        side.unlink(missing_ok=True)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"删除失败：{e}") from e
    logger.info("已删除推理结果：%s", filename)
    return {"ok": True, "filename": filename}


# ---------- 静态页面（http://127.0.0.1:8000）----------
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


# ---------- 启动 ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
