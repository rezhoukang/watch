"""YOLOv8 封装：图片检测 / 视频检测 / 模型训练

对外只暴露简单函数，模型实例按权重名缓存，避免重复加载。
ultralytics / torch 采用懒加载：即使依赖未装好，Web 服务也能正常启动。
"""
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"            # 唯一的权重目录（含自动下载的缓存）
RESULTS_DIR = BASE_DIR / "data" / "results"  # 检测结果图
VIDEO_DIR = BASE_DIR / "data" / "videos"     # 上传的视频
OUTPUT_DIR = BASE_DIR / "data" / "outputs"   # 检测结果视频
DATASETS_DIR = BASE_DIR / "data" / "datasets"  # 训练数据集
RUNS_DIR = BASE_DIR / "data" / "runs"        # 训练输出

DEFAULT_WEIGHTS = "yolov8s-worldv2.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

# ---------- Windows 下让 OpenCV 能找到 OpenH264（H.264 编码）----------
# 浏览器 <video> 只认 H.264 / VP9，OpenCV 默认的 mp4v 是 MPEG-4 Part 2 播不了。
# OpenCV 的 FFmpeg 加载 avc1 时按硬编码文件名找 openh264-2.5.0-win64.dll，
# 搜索顺序含「当前工作目录」但 os.add_dll_directory 对它无效（实测），
# 所以把 DLL 从 data/openh264/ 复制到工作目录（幂等，已存在则跳过）。
_OPENH264_NAME = "openh264-2.5.0-win64.dll"
if sys.platform == "win32":
    _src_dll = BASE_DIR / "data" / "openh264" / _OPENH264_NAME
    try:
        if _src_dll.is_file() and not (Path.cwd() / _OPENH264_NAME).is_file():
            import shutil

            shutil.copyfile(_src_dll, Path.cwd() / _OPENH264_NAME)
    except Exception:  # noqa: BLE001
        pass

# 线程安全
_lock = threading.RLock()
# 用 Any 而非 object：ultralytics 的类型标注不完整，标成 object 会让
# Pylance 把 model.predict / results[0] / .boxes 等全判为不可访问
_model_cache: dict[str, Any] = {}
# World 模型当前生效的类别（文本提示）缓存 —— 类别没变就不重复 set_classes
_model_classes: dict[str, tuple[str, ...]] = {}

# 后台任务表（训练 / 视频处理）
JOBS: dict[str, dict] = {}


def cancel_job(job_id: str) -> bool:
    """请求取消一个运行中的任务。返回 False 表示任务不存在或已结束。"""
    job = JOBS.get(job_id)
    if job is None:
        return False
    if job.get("status") == "running":
        job["cancel"] = True
        job["message"] = "正在取消…"
        return True
    return False


def _cancelled(job: dict) -> bool:
    """后台线程里检查取消标志"""
    return bool(job.get("cancel"))


class _JobCancelled(Exception):
    """后台任务被取消时抛出，让外层 except 能识别并标记为 cancelled"""

    pass


def _ensure_dirs() -> None:
    for d in (MODELS_DIR, RESULTS_DIR, VIDEO_DIR, OUTPUT_DIR, DATASETS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


_unified_weights_dir = False


def _unify_weights_dir() -> None:
    """把 ultralytics 的自动下载目录从默认的 weights/ 改到 models/。

    不改的话项目里会出现两个权重目录：
      - models/            本项目的业务权重目录（后端只扫这里生成下拉框）
      - weights/           ultralytics 默认目录，CLIP 文本编码器（337MB）会下到这里
    注意 ultralytics.nn.text_model 是 import 时绑定 WEIGHTS_DIR 的，
    必须连它一起改，否则 CLIP 会在新位置重新下载一遍。
    CLIP 落在 models/clip/ 子目录，而模型清单只扫 models/*.pt（不递归），不会互相污染。
    """
    global _unified_weights_dir
    if _unified_weights_dir:
        return
    try:
        import ultralytics.nn.text_model as _text_model
        import ultralytics.utils as _utils

        _utils.SETTINGS["weights_dir"] = str(MODELS_DIR)
        _utils.WEIGHTS_DIR = MODELS_DIR
        # text_model 里是 `from ultralytics.utils import WEIGHTS_DIR`，属于导入来的
        # 名字而非该模块自身的导出，所以类型检查器不认这个属性（运行时确实存在）。
        _text_model.WEIGHTS_DIR = MODELS_DIR  # type: ignore[attr-defined]
    except ImportError:
        pass  # 依赖未装好时静默跳过，Web 服务照常启动
    _unified_weights_dir = True


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


# ---------- YOLO-World（开放词表）支持 ----------
def is_world_weights(weights: str) -> bool:
    """是否为 YOLO-World 权重（按文件名判断，如 yolov8s-worldv2.pt）"""
    return "world" in Path(weights).name.lower()


def clip_available() -> bool:
    """YOLO-World 靠 CLIP 做文本编码，检查该依赖是否已安装"""
    import importlib.util

    return importlib.util.find_spec("clip") is not None


def world_ready(weights: str) -> tuple[bool, str]:
    """判断指定权重当前能否使用，返回 (是否可用, 说明)"""
    if not is_world_weights(weights):
        return True, "普通 YOLO（类别固定为权重内置类别）"
    if not clip_available():
        return False, '缺少 clip 依赖，请执行：pip install "git+https://github.com/ultralytics/CLIP.git"'
    return True, "YOLO-World 开放词表（类别可用文字随意指定）"


def get_model(weights: str = DEFAULT_WEIGHTS, classes: list[str] | None = None) -> Any:
    """加载并缓存模型（权重不存在会自动下载到 models/ 目录）

    - 文件名含 world → 走 YOLO-World（开放词表），可用 classes 传任意类别文字
    - 其它权重 → 走普通 YOLO（类别固定为权重内置类别，classes 被忽略）
    - classes 与上次相同则跳过 set_classes（文本编码较慢，能省则省）

    返回 Any：ultralytics 的类型标注不完整，强类型反而会误报。
    """
    _ensure_dirs()
    _unify_weights_dir()
    with _lock:
        if weights not in _model_cache:
            # 只认文件名，防止路径穿越；下载缓存放 models/
            name = Path(weights).name
            local = MODELS_DIR / name
            # 始终传绝对路径：ultralytics 拼下载 URL 时只取文件名，
            # 但落盘用的是传入路径——传裸文件名会把新下载的权重丢到当前工作目录
            path = str(local)

            if is_world_weights(weights):
                from ultralytics import YOLOWorld

                _model_cache[weights] = YOLOWorld(path)
            else:
                from ultralytics import YOLO

                _model_cache[weights] = YOLO(path)

            _model_classes.pop(weights, None)  # 新实例，类别状态重置

        model = _model_cache[weights]

        # World 模型：按需切换类别（文字 → 类别嵌入）
        if classes and is_world_weights(weights):
            key = tuple(classes)
            if _model_classes.get(weights) != key:
                model.set_classes(list(key))
                _model_classes[weights] = key

        return model


def current_classes(weights: str) -> list[str] | None:
    """当前模型生效的 World 类别（没设过则返回 None，表示用权重内置类别）"""
    key = _model_classes.get(weights)
    return list(key) if key else None


def clear_model_cache() -> int:
    with _lock:
        n = len(_model_cache)
        _model_cache.clear()
        return n


# ---------- 图片检测 ----------
def _clamp_imgsz(imgsz: int) -> int:
    """推理尺寸钳制到 320~1920，并对齐 32 的倍数（YOLO 下采样步长）"""
    v = max(320, min(int(imgsz), 1920))
    return int(round(v / 32) * 32)


def detect_image(
    src: Path,
    weights: str = DEFAULT_WEIGHTS,
    conf: float = 0.25,
    classes: list[str] | None = None,
    imgsz: int = 640,
) -> dict:
    """检测单张图片，保存带框结果，返回检测详情

    classes: 仅对 World 模型生效；为空则使用权重内置类别
    imgsz:   推理分辨率（会钳制到 320~1920 并对齐 32 的倍数）
    """
    _ensure_dirs()
    import cv2

    size = _clamp_imgsz(imgsz)
    model = get_model(weights, classes=classes)
    results = model.predict(
        source=str(src), conf=conf, imgsz=size, device=device_arg(), verbose=False
    )
    r = results[0]

    # 保存带框结果图
    # cv2.imwrite 在 Windows 下遇到中文路径会静默失败（内部用窄字符 fopen），
    # 所以走 imencode + tofile 写字节，保证「东教室阶梯.png」这类文件名也能存。
    out_name = f"{src.stem}_det{src.suffix}"
    out_path = RESULTS_DIR / out_name
    ok, buf = cv2.imencode(out_path.suffix, r.plot())
    if not ok:
        raise RuntimeError(f"结果图编码失败：{out_name}")
    buf.tofile(str(out_path))

    names = r.names
    orig_h, orig_w = r.orig_shape
    detections = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = (round(float(v), 2) for v in box.xyxy[0].tolist())
        detections.append(
            {
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": round(float(box.conf[0]), 4),
                # 左上右下（原图像素）——画框、裁剪用
                "bbox": [x1, y1, x2, y2],
                # 中心点 + 宽高（原图像素）——由 bbox 反算，保证两者严格自洽
                "bbox_cxcywh": [
                    round((x1 + x2) / 2, 2),
                    round((y1 + y2) / 2, 2),
                    round(x2 - x1, 2),
                    round(y2 - y1, 2),
                ],
            }
        )

    # 分类统计
    summary: dict[str, int] = {}
    for d in detections:
        summary[d["class_name"]] = summary.get(d["class_name"], 0) + 1

    # 当前生效的类别清单（给前端展示用）
    if isinstance(names, dict):
        classes_used = [str(v) for _, v in sorted(names.items())]
    else:
        classes_used = [str(v) for v in names]

    return {
        "source": src.name,
        "result": out_name,
        "result_url": f"/results/{out_name}",
        "image_width": int(orig_w),
        "image_height": int(orig_h),
        "imgsz": size,
        "count": len(detections),
        "summary": summary,
        "detections": detections,
        "mode": "world" if is_world_weights(weights) else "yolo",
        "model": Path(weights).name,
        "classes_used": classes_used,
    }


# ---------- 视频检测（后台任务）----------
def video_info(src: Path) -> dict | None:
    """读取视频元信息（前端时间轴用）。读取失败返回 None"""
    try:
        import cv2

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {
            "duration": round(total / fps, 2) if total else 0.0,
            "width": w,
            "height": h,
            "fps": round(fps, 3),
            "frames": total,
        }
    except Exception:  # noqa: BLE001
        return None


def _annotate_frame(
    frame, r: Any, render: str = "box", line_width: int = 2
) -> Any:
    """按选定的显示形式把检测结果画到帧上，返回可写入视频的 BGR 图

    render:
      - "box" 默认：框 + 类别标签
      - "dot" 点模式：只在每个目标中心画小圆点 + 迷你标签，
        适合小目标/密集场景，不会糊成一团（框太粗盖住人）
    line_width: 框线粗细（box 模式生效）
    """
    import cv2
    from ultralytics.utils.plotting import Annotator, Colors

    # 类别颜色与模型一致（按 class_id 取色）
    colors = Colors()
    annotator = Annotator(frame, line_width=line_width)
    names = r.names

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        color = colors(cls_id, bgr=True)
        name = names.get(cls_id, str(cls_id))
        label = f"{name} {conf:.2f}"

        if render == "dot":
            # 中心小圆点（半径 3），旁边挂迷你标签，不画框
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 3, color, -1)  # type: ignore[attr-defined]
            cv2.putText(
                frame,
                label,
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore[attr-defined]
                0.4,
                color,
                1,
                cv2.LINE_AA,  # type: ignore[attr-defined]
            )
        else:
            annotator.box_label((x1, y1, x2, y2), label, color=color)

    return frame


def start_video_job(
    src: Path,
    weights: str = DEFAULT_WEIGHTS,
    conf: float = 0.25,
    stride: int = 1,
    classes: list[str] | None = None,
    start: float = 0.0,
    end: float | None = None,
    render: str = "box",
    line_width: int = 2,
) -> str:
    """启动视频检测后台任务，返回 job_id

    stride: 每 N 帧检测一次（>1 时中间帧复用上次结果，可提速）
    classes: 仅对 World 模型生效
    start/end: 只检测 [start, end) 秒区间（默认全片）；输出视频只含该区间
    render: "box" 框 / "dot" 中心点（密集小目标用点更清晰）
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
        args=(job_id, src, weights, conf, stride, classes, start, end, render, line_width),
        daemon=True,
    ).start()
    return job_id


def _run_video_job(
    job_id: str,
    src: Path,
    weights: str,
    conf: float,
    stride: int,
    classes: list[str] | None = None,
    start: float = 0.0,
    end: float | None = None,
    render: str = "box",
    line_width: int = 2,
) -> None:
    job = JOBS[job_id]
    cap = None
    writer = None
    out_path = None
    try:
        import cv2

        model = get_model(weights, classes=classes)
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError("无法打开视频文件")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # 时间范围 → 帧范围（越界自动收窄，空区间直接报错）
        start = max(0.0, float(start))
        end = float(end) if end is not None else total / fps
        end = min(end, total / fps) if total else end
        f0 = int(round(start * fps))
        f1 = int(round(end * fps))
        if f1 <= f0:
            raise RuntimeError(f"无效的时间范围：{start:.1f}s ~ {end:.1f}s（视频总长 {total / fps:.1f}s）")

        # 输出文件名带上时间范围，避免覆盖同名的全片结果
        if f0 > 0 or f1 < total:
            tag = f"_{int(start)}s-{int(end)}s"
        else:
            tag = ""
        # 点模式在文件名上标记，避免和框模式互相覆盖
        render_tag = "" if render == "box" else "_dot"
        out_path = OUTPUT_DIR / f"{src.stem}_det{render_tag}{tag}.mp4"

        # H.264（avc1）让浏览器能直接播放；mp4v 是 MPEG-4 Part 2，<video> 不支持。
        # 若 OpenH264 DLL 没加载成功会回退 mp4v（仍能生成文件，只是网页不能预览）。
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"avc1"),  # type: ignore[attr-defined]
            fps,
            (w, h),
        )
        if not writer.isOpened():
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
                fps,
                (w, h),
            )

        # 定位到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

        dev = device_arg()
        idx = 0
        last_annotated = None
        while idx < f1 - f0:
            if _cancelled(job):
                raise _JobCancelled()
            ok, frame = cap.read()
            if not ok:
                break
            # 按 stride 抽帧检测，其余帧复用上一帧结果
            if idx % stride == 0:
                r = model.predict(source=frame, conf=conf, device=dev, verbose=False)[0]
                last_annotated = _annotate_frame(
                    frame.copy(), r, render=render, line_width=line_width
                )
            writer.write(last_annotated if last_annotated is not None else frame)

            idx += 1
            job["progress"] = idx
            job["message"] = f"{idx}/{f1 - f0} 帧"

        cap.release()
        writer.release()

        job["total"] = f1 - f0
        job["status"] = "done"
        job["message"] = f"完成，共 {idx} 帧（{start:.0f}s ~ {end:.0f}s）"
        job["result_url"] = f"/outputs/{out_path.name}"
    except _JobCancelled:
        # 取消：尽量释放资源、删掉半成品，标记为 cancelled
        try:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
        except Exception:  # noqa: BLE001
            pass
        try:
            if out_path is not None:
                out_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        job["status"] = "cancelled"
        job["message"] = "已取消"
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
            # 取消请求 → 抛异常中断训练（ultralytics 会在本轮结束后停下）
            if _cancelled(job):
                raise _JobCancelled()

        model.add_callback("on_train_epoch_end", on_epoch_end)

        try:
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
        except _JobCancelled:
            job["status"] = "cancelled"
            job["message"] = "已取消"
            return

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
    """训练起点权重：统一指向 models/，文件不存在时由 ultralytics 下到那里"""
    _unify_weights_dir()
    return str(MODELS_DIR / Path(weights).name)


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
