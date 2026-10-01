"""YOLOv8 封装：图片检测 / 视频检测 / 模型训练

对外只暴露简单函数，模型实例按权重名缓存，避免重复加载。
ultralytics / torch 采用懒加载：即使依赖未装好，Web 服务也能正常启动。
"""
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"            # 权重文件（首次使用自动下载）
RESULTS_DIR = BASE_DIR / "data" / "results"  # 检测结果图
VIDEO_DIR = BASE_DIR / "data" / "videos"     # 上传的视频
OUTPUT_DIR = BASE_DIR / "data" / "outputs"   # 检测结果视频
DATASETS_DIR = BASE_DIR / "data" / "datasets"  # 训练数据集
RUNS_DIR = BASE_DIR / "data" / "runs"        # 训练输出

DEFAULT_WEIGHTS = "yolov8n.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

# 线程安全
_lock = threading.RLock()
# 用 Any 而非 object：ultralytics 的类型标注不完整，标成 object 会让
# Pylance 把 model.predict / results[0] / .boxes 等全判为不可访问
_model_cache: dict[str, Any] = {}

# 后台任务表（训练 / 视频处理）
JOBS: dict[str, dict] = {}


def _ensure_dirs() -> None:
    for d in (MODELS_DIR, RESULTS_DIR, VIDEO_DIR, OUTPUT_DIR, DATASETS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def yolo_available() -> tuple[bool, str]:
    """检查 YOLO 依赖是否可用，返回 (是否可用, 说明)"""
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:
        return False, f"未安装 ultralytics：{e}"
    try:
        import torch

        if torch.cuda.is_available():
            return True, f"GPU 可用：{torch.cuda.get_device_name(0)}"
        return True, "仅 CPU 可用（未检测到 CUDA）"
    except ImportError as e:
        return False, f"未安装 torch：{e}"


def device_arg() -> "int | str":
    """返回 ultralytics 的 device 参数：有 GPU 用 0，否则 cpu"""
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_model(weights: str = DEFAULT_WEIGHTS) -> Any:
    """加载并缓存模型（权重不存在会自动下载到 models/ 目录）

    返回 Any：ultralytics 的 YOLO 类型标注不完整，强类型反而会误报。
    """
    _ensure_dirs()
    with _lock:
        if weights in _model_cache:
            return _model_cache[weights]

        from ultralytics import YOLO

        # 只认文件名，防止路径穿越；下载缓存放 models/
        name = Path(weights).name
        local = MODELS_DIR / name
        model = YOLO(str(local) if local.exists() else name)
        _model_cache[weights] = model
        return model


def clear_model_cache() -> int:
    with _lock:
        n = len(_model_cache)
        _model_cache.clear()
        return n


# ---------- 图片检测 ----------
def detect_image(
    src: Path,
    weights: str = DEFAULT_WEIGHTS,
    conf: float = 0.25,
) -> dict:
    """检测单张图片，保存带框结果，返回检测详情"""
    _ensure_dirs()
    import cv2

    model = get_model(weights)
    results = model.predict(source=str(src), conf=conf, device=device_arg(), verbose=False)
    r = results[0]

    # 保存带框结果图
    out_name = f"{src.stem}_det{src.suffix}"
    out_path = RESULTS_DIR / out_name
    cv2.imwrite(str(out_path), r.plot())

    names = r.names
    detections = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        detections.append(
            {
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(float(v), 2) for v in box.xyxy[0].tolist()],
            }
        )

    # 分类统计
    summary: dict[str, int] = {}
    for d in detections:
        summary[d["class_name"]] = summary.get(d["class_name"], 0) + 1

    return {
        "source": src.name,
        "result": out_name,
        "result_url": f"/results/{out_name}",
        "count": len(detections),
        "summary": summary,
        "detections": detections,
    }


# ---------- 视频检测（后台任务）----------
def start_video_job(
    src: Path,
    weights: str = DEFAULT_WEIGHTS,
    conf: float = 0.25,
    stride: int = 1,
) -> str:
    """启动视频检测后台任务，返回 job_id

    stride: 每 N 帧检测一次（>1 时中间帧复用上次结果，可提速）
    """
    _ensure_dirs()
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id,
        "type": "video",
        "status": "running",
        "progress": 0,
        "total": 0,
        "message": "准备中…",
        "result_url": None,
    }
    threading.Thread(
        target=_run_video_job,
        args=(job_id, src, weights, conf, stride),
        daemon=True,
    ).start()
    return job_id


def _run_video_job(job_id: str, src: Path, weights: str, conf: float, stride: int) -> None:
    job = JOBS[job_id]
    try:
        import cv2

        model = get_model(weights)
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError("无法打开视频文件")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        job["total"] = total

        out_path = OUTPUT_DIR / f"{src.stem}_det.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]  # cv2 类型存根缺这个属性
            fps,
            (w, h),
        )

        dev = device_arg()
        idx = 0
        last_annotated = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            # 按 stride 抽帧检测，其余帧复用上一帧结果
            if (idx - 1) % stride == 0:
                r = model.predict(source=frame, conf=conf, device=dev, verbose=False)[0]
                last_annotated = r.plot()
            writer.write(last_annotated if last_annotated is not None else frame)

            if total:
                job["progress"] = idx
                job["message"] = f"{idx}/{total} 帧"

        cap.release()
        writer.release()

        job["status"] = "done"
        job["message"] = f"完成，共 {idx} 帧"
        job["result_url"] = f"/outputs/{out_path.name}"
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["message"] = f"处理失败：{e}"


# ---------- 训练（后台任务）----------
def list_datasets() -> list[dict]:
    """列出 data/datasets/ 下的数据集及其基本信息"""
    _ensure_dirs()
    out = []
    for d in sorted(DATASETS_DIR.iterdir()):
        if not d.is_dir():
            continue
        # 统计图片数量（支持 images/train 或 images/ 两种结构）
        img_count = 0
        for sub in ("images/train", "images/val", "images"):
            p = d / sub
            if p.is_dir():
                img_count += sum(
                    1 for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
                )
        out.append(
            {
                "name": d.name,
                "images": img_count,
                "has_yaml": (d / "data.yaml").exists(),
            }
        )
    return out


def start_train_job(
    dataset: str,
    weights: str = DEFAULT_WEIGHTS,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    lr0: float = 0.01,
) -> str:
    """启动训练后台任务，返回 job_id"""
    _ensure_dirs()
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id,
        "type": "train",
        "status": "running",
        "progress": 0,
        "total": epochs,
        "message": "准备中…",
        "result_url": None,
        "best_weights": None,
    }
    threading.Thread(
        target=_run_train_job,
        args=(job_id, dataset, weights, epochs, imgsz, batch, lr0),
        daemon=True,
    ).start()
    return job_id


def _run_train_job(
    job_id: str,
    dataset: str,
    weights: str,
    epochs: int,
    imgsz: int,
    batch: int,
    lr0: float,
) -> None:
    job = JOBS[job_id]
    try:
        from ultralytics import YOLO

        ds_dir = DATASETS_DIR / dataset
        if not ds_dir.is_dir():
            raise RuntimeError(f"数据集不存在：{dataset}")

        data_yaml = _prepare_data_yaml(ds_dir)
        model = YOLO(_resolve_weights(weights))

        def on_epoch_end(trainer) -> None:
            job["progress"] = trainer.epoch + 1
            job["message"] = f"第 {trainer.epoch + 1}/{trainer.epochs} 轮"
            job["metrics"] = _extract_metrics(trainer)

        model.add_callback("on_train_epoch_end", on_epoch_end)

        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            device=device_arg(),
            project=str(RUNS_DIR),
            name=job_id,
            exist_ok=True,
            plots=True,
        )

        best = RUNS_DIR / job_id / "weights" / "best.pt"
        job["best_weights"] = str(best) if best.exists() else None
        job["status"] = "done"
        job["message"] = f"训练完成（{epochs} 轮）"
        if best.exists():
            # 下载地址：/models/{run}/{filename}
            job["result_url"] = f"/models/{job_id}/{best.name}"
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["message"] = f"训练失败：{e}"


def _resolve_weights(weights: str) -> str:
    """训练起点权重：models/ 里有的用本地，否则用官方名（自动下载）"""
    name = Path(weights).name
    local = MODELS_DIR / name
    return str(local) if local.exists() else name


def _extract_metrics(trainer) -> dict:
    """从 trainer 里提取常用指标（失败则返回空）"""
    try:
        m = getattr(trainer, "metrics", None) or {}
        out = {}
        for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "train/box_loss", "train/cls_loss"):
            if k in m:
                out[k.split("/")[-1]] = round(float(m[k]), 4)
        return out
    except Exception:  # noqa: BLE001
        return {}


def _prepare_data_yaml(ds_dir: Path) -> Path:
    """若数据集没有 data.yaml，根据 images/labels 结构自动生成

    支持两种结构：
      A. <ds>/images/{train,val}、<ds>/labels/{train,val}  （YOLO 标准）
      B. <ds>/images、<ds>/labels                          （单目录，全部当训练集）
    """
    yaml_path = ds_dir / "data.yaml"
    if yaml_path.exists():
        return yaml_path

    lines = [f"path: {ds_dir.as_posix()}", "train: images/train"]
    if (ds_dir / "images" / "val").is_dir():
        lines.append("val: images/val")
    else:
        lines.append("val: images/train")

    names = _read_classes(ds_dir)
    lines.append(f"nc: {len(names)}")
    lines.append("names:")
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def _read_classes(ds_dir: Path) -> list[str]:
    """从 classes.txt 读取类别名；没有则用 class_0, class_1…"""
    classes_file = ds_dir / "classes.txt"
    if classes_file.exists():
        names = [
            ln.strip()
            for ln in classes_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if names:
            return names

    # 从标注里推断最大类别 id
    max_id = -1
    for lbl_dir in (ds_dir / "labels" / "train", ds_dir / "labels"):
        if not lbl_dir.is_dir():
            continue
        for f in lbl_dir.glob("*.txt"):
            for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = ln.split()
                if parts and parts[0].isdigit():
                    max_id = max(max_id, int(parts[0]))
    return [f"class_{i}" for i in range(max_id + 1)] if max_id >= 0 else ["object"]
