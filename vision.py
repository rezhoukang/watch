"""视觉模型封装：YOLO 图片/视频/训练、RT-DETRv4 图片人头计数

对外只暴露简单函数，模型实例按权重名缓存，避免重复加载。
ultralytics / torch 采用懒加载：即使依赖未装好，Web 服务也能正常启动。
"""
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import rtdetrv4
from dataset_store import (
    list_datasets,
    dataset_detail,
    resolve_annotate_image,
    split_of,
    find_dataset_dir,
    dataset_classes,
    _read_yaml_names,
    _infer_classes,
    append_dataset_classes,
    write_dataset_classes,
    read_labels_for_image,
    save_labels_for_image,
)

from project_paths import (
    BASE_DIR, MODELS_DIR, UPLOAD_DIR, RESULTS_DIR, VIDEO_DIR,
    OUTPUT_DIR, DATASETS_DIR, RUNS_DIR, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS,
)

DEFAULT_WEIGHTS = "medium.pt"

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
    if rtdetrv4.is_weights(weights):
        raise ValueError("RT-DETRv4 当前仅支持图片人数统计，请使用 detect_image")
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
        return n + rtdetrv4.clear_cache()


# ---------- 图片推理 ----------
def _clamp_imgsz(imgsz: int) -> int:
    """推理尺寸钳制到 320~1920，并对齐 32 的倍数（YOLO 下采样步长）"""
    v = max(320, min(int(imgsz), 1920))
    return int(round(v / 32) * 32)


def detect_image(
    src: Path,
    weights: str = DEFAULT_WEIGHTS,
    conf: float = 0.38,
    classes: list[str] | None = None,
    imgsz: int = 640,
    save_result: bool = True,
) -> dict:
    """推理单张图片，保存带框结果，返回推理详情

    classes: 仅对 World 模型生效；RT-DETRv4 固定只输出 head
    imgsz:   推理分辨率（会钳制到 320~1920 并对齐 32 的倍数）
    save_result: False 时不写结果图（预标注用，避免污染 results/）
    """
    _ensure_dirs()
    size = _clamp_imgsz(imgsz)
    if rtdetrv4.is_weights(weights):
        return rtdetrv4.detect_image(src, conf, size, RESULTS_DIR, save_result=save_result)
    import cv2

    model = get_model(weights, classes=classes)
    results = model.predict(
        source=str(src), conf=conf, imgsz=size, device=device_arg(), verbose=False
    )
    r = results[0]

    # 保存带框结果图
    # cv2.imwrite 在 Windows 下遇到中文路径会静默失败（内部用窄字符 fopen），
    # 所以走 imencode + tofile 写字节，保证「东教室阶梯.png」这类文件名也能存。
    out_name = ""
    if save_result:
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
        "result": out_name if save_result else "",
        "result_url": f"/results/{out_name}" if save_result else "",
        "image_width": int(orig_w),
        "image_height": int(orig_h),
        "imgsz": size,
        "count": len(detections),
        "summary": summary,
        "detections": detections,
        "mode": "world" if is_world_weights(weights) else "yolo",
        "weight": Path(weights).name,
        "classes_used": classes_used,
    }


# ---------- 视频推理（后台任务）----------
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
    """按选定的显示形式把推理结果画到帧上，返回可写入视频的 BGR 图

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


def _collect_best_frame(frame, frame_index: int, count: int) -> dict:
    """收集一帧，用于「图预览」：选检出目标数最多的那一帧（信息量最大）

    简化版：只记录帧本身（前端拿到后会重新走图片推理，不需要这里的坐标）。
    """
    return {
        "frame": frame.copy(),  # 必须 copy：cap.read 可能复用缓冲区
        "frame_index": frame_index,
        "count": count,
    }


def _save_best_frame(best: dict, src_stem: str) -> str | None:
    """把「召回率最高帧」保存为待推理图片（data/uploads/），返回文件名

    src_stem: 原始视频名（不含 _det 后缀），这样最佳帧能被 list_videos 匹配回输出视频。
    用户点「图预览」后，前端直接拿这个文件名走图片推理（自动切到推理结果页）。
    后端只存图，不做任何渲染。

    写入前先删掉同一视频上次提取的最佳帧，避免 uploads/ 里不断堆积
    （文件名含帧号，不复用旧名；但一个源视频只需要保留最新的一张）。
    """
    import cv2

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{src_stem}_best_{best['frame_index']}.jpg"
    # 清理该源视频之前提取的最佳帧（不同帧号 = 不同文件名，会越积越多）
    for old in UPLOAD_DIR.glob(f"{src_stem}_best_*.jpg"):
        try:
            old.unlink(missing_ok=True)
        except OSError:
            pass
    dest = UPLOAD_DIR / name
    # imencode + tofile 写字节，绕开 Windows 中文路径的窄字符 fopen
    ok, buf = cv2.imencode(".jpg", best["frame"])
    if not ok:
        return None
    buf.tofile(str(dest))
    return name


def _count_class(r: Any, name_hint: str = "head") -> int:
    """统计推理结果里某类（默认 head）的检出数。

    优先按类别名匹配（不区分大小写，支持子串如 'head' / 'person'）；
    匹配不到时返回 0（该权重没有这类）。
    """
    names = r.names
    hint = name_hint.lower()
    target_ids: list[int] = []
    if isinstance(names, dict):
        for cid, cname in names.items():
            if hint in str(cname).lower():
                target_ids.append(int(cid))
    else:
        for cid, cname in enumerate(names):
            if hint in str(cname).lower():
                target_ids.append(cid)
    if not target_ids:
        return 0
    n = 0
    for box in r.boxes:
        if int(box.cls[0]) in target_ids:
            n += 1
    return n


def _save_head_chart(
    head_counts: list, out_path: Path, fps: float = 25.0, start_frame: int = 0, metadata: dict | None = None
) -> str | None:
    """把每帧 head 计数存为 json（data/outputs/frames/<视频stem>.json），返回文件名

    前端据此画「人数折线图」。数据为 [[frame_idx, count], ...]（帧号为原视频绝对帧号）。
    fps / start_frame 用来把帧号换算成时间（秒）显示在 X 轴上。
    """
    import json

    frames_dir = OUTPUT_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    name = f"{out_path.stem}.json"
    payload = {
        "head_counts": head_counts,
        "source": out_path.name,
        "fps": round(float(fps), 3) if fps else 25.0,
        "start_frame": int(start_frame),
    }
    if metadata:
        payload.update(metadata)
    (frames_dir / name).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return name


def start_video_job(
    src: Path,
    weights: str = DEFAULT_WEIGHTS,
    conf: float = 0.38,
    stride: int = 1,
    classes: list[str] | None = None,
    start: float = 0.0,
    end: float | None = None,
    render: str = "box",
    line_width: int = 2,
    head_detection: bool = False,
    rtdetr_base: bool = False,
    merge_iou: float = 0.0,
    merge_confidence: float = 0.0,
) -> str:
    """启动视频推理后台任务，返回 job_id

    stride: 每 N 帧推理一次（>1 时中间帧复用上次结果，可提速）
    classes: 仅对 World 模型生效
    start/end: 只推理 [start, end) 秒区间（默认全片）；输出视频只含该区间
    render: "box" 框 / "dot" 中心点（密集小目标用点更清晰）
    """
    _ensure_dirs()
    head_fusion_mod: Any = None
    if head_detection:
        import head_fusion

        head_fusion_mod = head_fusion
        weights, conf, classes = head_fusion.BASE_WEIGHTS, head_fusion.BASE_CONFIDENCE, None
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id,
        "type": "video",
        "status": "running",
        "progress": 0,
        "total": 0,
        "message": "准备中…",
        "result_url": None,
        "head_counts": [],   # 每帧 head 类检出数（折线图数据），形如 [(frame_idx, count)]
        "head_detection": head_detection,
        "rtdetr_base": head_detection and rtdetr_base,
        "parameters": head_fusion_mod.parameters(rtdetr_base, merge_iou, merge_confidence) if head_detection else {"weights": weights, "confidence": conf},
    }
    threading.Thread(
        target=_run_video_job,
        args=(job_id, src, weights, conf, stride, classes, start, end, render, line_width, head_detection, rtdetr_base, merge_iou, merge_confidence),
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
    head_detection: bool = False,
    rtdetr_base: bool = False,
    merge_iou: float = 0.0,
    merge_confidence: float = 0.0,
) -> None:
    job = JOBS[job_id]
    cap = None
    writer = None
    out_path = None
    detection_file = None
    detection_path = None
    try:
        import cv2
        import json
        import head_fusion

        standalone_rtdetr = rtdetrv4.is_weights(weights) and not head_detection
        model = None if standalone_rtdetr else get_model(weights, classes=classes)
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
        render_tag = (("_head_rtdetrbase" if rtdetr_base else "_head") if head_detection else "_rtdetrv4" if standalone_rtdetr else "") + ("" if render == "box" else "_dot")
        if head_detection and merge_iou > 0:
            render_tag += f"_iou{merge_iou:g}"
            if merge_confidence > 0:
                render_tag += f"_gate{merge_confidence:g}"
        out_path = OUTPUT_DIR / f"{src.stem}_det{render_tag}{tag}.mp4"
        if head_detection:
            detection_path = OUTPUT_DIR / "detections" / f"{out_path.stem}.jsonl"
            detection_path.parent.mkdir(parents=True, exist_ok=True)
            detection_file = detection_path.open("w", encoding="utf-8")
            detection_file.write(json.dumps({"parameters": head_fusion.parameters(rtdetr_base, merge_iou, merge_confidence), "fps": fps,
                                            "stride": stride, "start_frame": f0}, ensure_ascii=False) + "\n")

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
        best = None          # 召回率最高帧（目标数最多）
        while idx < f1 - f0:
            if _cancelled(job):
                raise _JobCancelled()
            ok, frame = cap.read()
            if not ok:
                break
            # 按 stride 抽帧推理，其余帧复用上一帧结果
            if idx % stride == 0:
                if head_detection:
                    assert detection_file is not None and detection_path is not None
                    fused = head_fusion.detect_frame(frame, model, rtdetr_base=rtdetr_base, merge_iou=merge_iou, merge_confidence=merge_confidence)
                    n = fused["count"]
                    head_count = sum(1 for det in fused["detections"] if "head" in det["class_name"].lower())
                    detection_file.write(json.dumps({"frame_index": f0 + idx, "time_seconds": (f0+idx)/fps,
                                                    "detections": fused["detections"], "fusion": fused["fusion"]},
                                                   ensure_ascii=False) + "\n")
                    last_annotated = head_fusion.annotate_frame(frame.copy(), fused, render, line_width)
                elif standalone_rtdetr:
                    from PIL import Image

                    detections, _ = rtdetrv4.detect_pil(Image.fromarray(frame[:, :, ::-1]), conf, 640)
                    for detection in detections:
                        detection["source_model"] = rtdetrv4.WEIGHTS_NAME
                    n = head_count = len(detections)
                    last_annotated = head_fusion.annotate_frame(
                        frame.copy(), {"detections": detections}, render, line_width
                    )
                else:
                    assert model is not None
                    r = model.predict(source=frame, conf=conf, device=dev, verbose=False)[0]
                    n, head_count = len(r.boxes), _count_class(r, "head")
                    last_annotated = _annotate_frame(
                        frame.copy(), r, render=render, line_width=line_width
                    )
                # 人数折线图数据：记录每帧 head 类检出数（stride 抽帧，中间帧不推理）
                job["head_counts"].append([f0 + idx, head_count])
                # 目标数最多的一帧 = 信息量最大（召回率代理指标）
                if best is None or n > best["count"]:
                    best = _collect_best_frame(frame, f0 + idx, n)
            writer.write(last_annotated if last_annotated is not None else frame)

            idx += 1
            job["progress"] = idx
            job["message"] = f"{idx}/{f1 - f0} 帧"

        cap.release()
        writer.release()
        if detection_file:
            assert detection_path is not None
            detection_file.close()
            job["detections_url"] = f"/outputs/detections/{detection_path.name}"

        # ⚠️ 顺序很重要：先把 chart_url / best_frame 等附加产物写好，最后才置 status="done"。
        # 否则前端轮询可能在 status 已 done、但 chart_url 还没写好的瞬间拿到结果，
        # 表现为「点预览没有折线图」。
        job["total"] = f1 - f0
        job["message"] = f"完成，共 {idx} 帧（{start:.0f}s ~ {end:.0f}s）"
        job["result_url"] = f"/outputs/{out_path.name}"
        # 人数折线图数据：存为 json 侧车（data/outputs/frames/），结果库可回看
        try:
            chart_name = _save_head_chart(job["head_counts"], out_path, fps=fps, start_frame=f0,
                                          metadata={"head_detection": head_detection, "parameters": job["parameters"],
                                                    "detections_url": job.get("detections_url")})
            if chart_name:
                job["chart_url"] = f"/outputs/frames/{chart_name}"
        except Exception:  # noqa: BLE001
            logger = __import__("logging").getLogger("watch")
            logger.exception("保存人数折线图数据失败")
        # 「图预览」：把召回率最高帧保存为待推理图片（data/uploads/），前端自动推理
        # 注意：只在 job 里记录文件名，不往 message 里拼——用户嫌完成信息太乱，
        # 最佳帧入口在结果库的「图预览」按钮（/api/videos 的 best_frame 字段）。
        if best is not None:
            try:
                best_name = _save_best_frame(best, src.stem)
                if best_name:
                    job["best_frame"] = best_name
            except Exception:  # noqa: BLE001
                logger = __import__("logging").getLogger("watch")
                logger.exception("保存视频最佳帧失败")

        # 最后一步才置 done：保证前端一旦看到 done，chart_url / best_frame 一定已就绪
        job["status"] = "done"
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
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if detection_file is not None:
            detection_file.close()
        if job["status"] != "done" and detection_path is not None:
            detection_path.unlink(missing_ok=True)


def prelabel_image(
    img_path: Path,
    weights: str = DEFAULT_WEIGHTS,
    conf: float = 0.38,
    classes: list[str] | None = None,
    imgsz: int = 640,
) -> list[dict]:
    """对单张图片做推理，返回可用作标注的框（原图像素 + 置信度）"""
    result = detect_image(
        img_path, weights=weights, conf=conf, classes=classes, imgsz=imgsz, save_result=False
    )
    return result["detections"]


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
    """确保 data.yaml 存在且合法，返回它的路径

    类别以 data.yaml 为唯一来源（不再有 classes.txt），所以这里不再"生成一份副本"，
    而是统一走 write_dataset_classes()：
      - 文件缺失 → 按目录结构 + label 反推生成
      - nc 与 names 不一致 → 修好（ultralytics 遇到不一致会直接抛 SyntaxError）
      - 缺 path/train/val → 补齐
    内容已一致时不会重写文件，也不会清标注缓存。
    """
    names = _read_yaml_names(ds_dir) or _infer_classes(ds_dir)
    write_dataset_classes(ds_dir, names)
    return ds_dir / "data.yaml"
