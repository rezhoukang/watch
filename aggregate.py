"""聚合操作：单图 目标检测 → 人脸识别（姓名贴到目标）→ 姿势检测与目标绑定。

三步互相独立、可分别开关：
1. 目标检测：按选定权重出目标框（不做姿势过滤、不做 ROI 关联——那是视频/标注页的策略）；
2. 人脸识别：整图 SCRFD 检出所有人脸 → 质量筛选 → ArcFace 提特征 → 1:N 比对，
   并把识别到的**姓名贴到最贴合的目标框**上（人脸中心落在哪个目标框内就归给谁，
   多个框都包含时取面积最小的；都不包含时退化为最近目标中心，超出容差就不归属）；
3. 姿势检测：Pose 权重出 17 点骨架，按选定策略（IoU / 中心包含 / 最近中心）与目标绑定。

结果图由后端画好（目标框 + 姓名标签 + 人脸框 + 骨架），前端直接展示即可。
"""
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import face_recognition
import pose_behaviors
import pose_support
import rtdetrv4
import vision
from project_paths import OUTPUT_DIR

# 结果图目录：独立子目录，避免和「图片推理」的 data/results 混在一起
RESULT_DIR = OUTPUT_DIR / "aggregate"

POSE_WEIGHTS = "yolov8m-pose.pt"
DEFAULT_POSE_CONFIDENCE = vision.DEFAULT_POSE_CONFIDENCE

# 姿势 ↔ 目标 的绑定策略（可选值，界面下拉框与之一致）
BIND_STRATEGIES = {
    "iou": "姿势框与目标框的 IoU 最大，且 ≥ 绑定 IoU",
    "contain": "目标框包含姿势框中心点（多个命中时取最小的目标框）",
    "nearest": "姿势框中心离目标框中心最近",
}
DEFAULT_BIND_STRATEGY = "iou"
DEFAULT_BIND_IOU = 0.3
# 人脸兜底归属：中心点不在任何目标框内时，允许吸附到 中心距离 ≤ 目标对角线 × 该系数 的目标
FACE_BIND_TOLERANCE = 0.5

# Windows 常见中文字体（找不到就退化为只写编号，避免姓名画成方块）
FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhl.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/arial.ttf",
)
_font_lock = threading.Lock()
_font_cache: dict[int, "ImageFont.FreeTypeFont | None"] = {}


# ---------------------------------------------------------------- 几何工具

def _area(box) -> float:
    return max(0.0, box[2]-box[0]) * max(0.0, box[3]-box[1])


def iou(a, b) -> float:
    overlap = max(0.0, min(a[2], b[2])-max(a[0], b[0])) * max(0.0, min(a[3], b[3])-max(a[1], b[1]))
    union = _area(a)+_area(b)-overlap
    return overlap/union if union > 0 else 0.0


def _center(box) -> tuple[float, float]:
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def _contains(box, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _distance(a, b) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


def _diagonal(box) -> float:
    return math.hypot(box[2]-box[0], box[3]-box[1])


# ---------------------------------------------------------------- 绑定

def bind_faces(faces: list[dict], targets: list[dict]) -> dict[str, dict]:
    """把每张人脸归到最贴合的目标上（识别到的姓名就贴在这个目标上）。

    策略：先找包住人脸中心的目标框，取其中面积最小的（最贴合）；都不包含时
    退化为最近目标中心，距离超过 目标对角线 × FACE_BIND_TOLERANCE 就不归属。
    返回 {face_id: {target_index, strategy, iou, distance}}。
    """
    bindings: dict[str, dict] = {}
    if not targets:
        return bindings
    for face in faces:
        box = face.get("bbox") or []
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            continue
        x, y = _center(box)
        inside = [(index, target) for index, target in enumerate(targets) if _contains(target["bbox"], x, y)]
        if inside:
            index, target = min(inside, key=lambda item: _area(item[1]["bbox"]))
            bindings[face["face_id"]] = {"target_index": index, "strategy": "center_inside",
                                         "iou": round(iou(box, target["bbox"]), 4), "distance": 0.0}
            continue
        index = min(range(len(targets)), key=lambda i: _distance(_center(targets[i]["bbox"]), (x, y)))
        target = targets[index]
        distance = _distance(_center(target["bbox"]), (x, y))
        diagonal = _diagonal(target["bbox"])
        if diagonal <= 0 or distance > diagonal*FACE_BIND_TOLERANCE:
            continue
        bindings[face["face_id"]] = {"target_index": index, "strategy": "nearest_center",
                                     "iou": round(iou(box, target["bbox"]), 4), "distance": round(distance, 2)}
    return bindings


def bind_poses(poses: list[dict], targets: list[dict], strategy: str = DEFAULT_BIND_STRATEGY,
               bind_iou: float = DEFAULT_BIND_IOU) -> dict[int, dict]:
    """按策略把姿势框绑到目标框上，返回 {姿势下标: {target_index, strategy, iou, distance}}。

    没绑上的姿势不进返回值——调用方据此把它们列为“未绑定”，而不是硬塞给某个目标。
    """
    if strategy not in BIND_STRATEGIES:
        raise ValueError(f"未知的绑定策略：{strategy}（可选：{'、'.join(BIND_STRATEGIES)}）")
    if not math.isfinite(bind_iou) or not 0 <= bind_iou <= 1:
        raise ValueError("绑定 IoU 阈值必须在 0 到 1 之间")
    bindings: dict[int, dict] = {}
    if not targets:
        return bindings
    for index, pose in enumerate(poses):
        box = pose.get("bbox") or []
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            continue
        if strategy == "nearest":
            x, y = _center(box)
            target = min(range(len(targets)), key=lambda i: _distance(_center(targets[i]["bbox"]), (x, y)))
            distance = _distance(_center(targets[target]["bbox"]), (x, y))
            bindings[index] = {"target_index": target, "strategy": strategy,
                               "iou": round(iou(box, targets[target]["bbox"]), 4), "distance": round(distance, 2)}
            continue
        if strategy == "contain":
            x, y = _center(box)
            inside = [(i, t) for i, t in enumerate(targets) if _contains(t["bbox"], x, y)]
            if not inside:
                continue
            target, chosen = min(inside, key=lambda item: _area(item[1]["bbox"]))
            bindings[index] = {"target_index": target, "strategy": strategy,
                               "iou": round(iou(box, chosen["bbox"]), 4), "distance": 0.0}
            continue
        # iou：取 IoU 最大的目标，低于阈值就判为未绑定（宁可漏绑也不乱绑）
        scored = [(iou(box, target["bbox"]), i) for i, target in enumerate(targets)]
        best, target = max(scored, key=lambda item: item[0])
        if best >= bind_iou:
            bindings[index] = {"target_index": target, "strategy": strategy,
                               "iou": round(best, 4), "distance": None}
    return bindings


# ---------------------------------------------------------------- 三步推理

def read_image(src: Path) -> np.ndarray:
    """读原图（中文路径安全）。"""
    image = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{src.name}")
    return image


def detect_targets(src: Path, weights: str, conf: float, classes: list[str] | None, imgsz: int) -> dict:
    """目标检测。不用 pose_support.filter_result，避免“无法判断”的人被静默丢掉。

    keypoints 仅在权重本身就是 Pose 权重时非空——这种权重下姿势检测可以直接复用，
    不必对同一张图再跑一次模型。
    """
    size = vision._clamp_imgsz(imgsz)
    if rtdetrv4.is_weights(weights):
        result = rtdetrv4.detect_image(src, conf, size, RESULT_DIR, save_result=False)
        detections = [{"class_id": int(d.get("class_id", 0)), "class_name": d["class_name"],
                       "confidence": round(float(d["confidence"]), 4), "bbox": [round(float(v), 2) for v in d["bbox"]]}
                      for d in result["detections"]]
        return {"detections": detections, "keypoints": [], "mode": "rtdetrv4", "imgsz": size,
                "classes_used": result.get("classes_used", []), "weight": Path(weights).name}

    model = vision.get_model(weights, classes=classes)
    predictions = model.predict(source=str(src), conf=conf, imgsz=size, device=vision.device_arg(), verbose=False)
    if len(predictions) != 1:
        raise RuntimeError("目标检测应返回一张图片的结果")
    result = predictions[0]
    names = result.names
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = (round(float(v), 2) for v in box.xyxy[0].tolist())
        detections.append({"class_id": cls_id, "class_name": str(names.get(cls_id, cls_id)),
                           "confidence": round(float(box.conf[0]), 4), "bbox": [x1, y1, x2, y2]})
    classes_used = [str(v) for _, v in sorted(names.items())] if isinstance(names, dict) else [str(v) for v in names]
    return {"detections": detections, "keypoints": pose_support.serialize_keypoints(result),
            "mode": "world" if vision.is_world_weights(weights) else "yolo", "imgsz": size,
            "classes_used": classes_used, "weight": Path(weights).name}


def detect_poses(src: Path, weights: str = POSE_WEIGHTS, conf: float = DEFAULT_POSE_CONFIDENCE,
                 imgsz: int = 640) -> list[dict]:
    """Pose 权重出人体框 + 骨架；行为按 pose_behaviors 默认阈值（低头 0.05 / 抬头 0.20）判定。"""
    model = vision.get_model(weights)
    predictions = model.predict(source=str(src), conf=conf, imgsz=vision._clamp_imgsz(imgsz),
                                device=vision.device_arg(), verbose=False)
    if len(predictions) != 1:
        raise RuntimeError("姿势检测应返回一张图片的结果")
    result = predictions[0]
    keypoints = pose_support.serialize_keypoints(result)
    if not keypoints:
        raise ValueError(f"权重 {Path(weights).name} 没有返回骨架，请选择 Pose 权重")
    height, width = result.orig_shape
    poses = []
    for index, (box, points) in enumerate(zip(result.boxes, keypoints), 1):
        x1, y1, x2, y2 = (round(float(v), 2) for v in box.xyxy[0].tolist())
        poses.append({"person_index": index, "bbox": [x1, y1, x2, y2],
                      "confidence": round(float(box.conf[0]), 4), "keypoints": points,
                      "behavior": pose_behaviors.classify(points, int(width), int(height))})
    return poses


# ---------------------------------------------------------------- 结果图

def _font(size: int = 18):
    """中文字体；取不到返回 None（调用方退化为只写编号，避免中文画成方块）。"""
    with _font_lock:
        if size not in _font_cache:
            font = None
            for candidate in FONT_CANDIDATES:
                try:
                    font = ImageFont.truetype(candidate, size)
                    break
                except OSError:
                    continue
            _font_cache[size] = font
        return _font_cache[size]


def _label(draw, x: float, y: float, text: str, font, background: str) -> None:
    """在框左上角画带底色的小标签，保证在复杂照片上也读得清。"""
    height = 16 if font is None else int(getattr(font, "size", 16)*1.4)
    width = max(28, len(text)*9) if font is None else int(draw.textlength(text, font=font))+8
    top = max(0.0, y-height)
    draw.rectangle((x, top, x+width, top+height), fill=background)
    draw.text((x+4, top+2), text, fill="#ffffff", font=font)


def render(image: np.ndarray, targets: list[dict], faces: list[dict], poses: list[dict]) -> Image.Image:
    """把三步结果画到一张图上：目标框 + 姓名标签、人脸框 + 五官点、骨架 + 行为标签。"""
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font = _font(18)

    for target in targets:
        name = target.get("name")
        color = "#00bb66" if name else "#2f6fed"
        draw.rectangle(target["bbox"], outline=color, width=3)
        if name:
            label = f"{name} {target.get('similarity')}"
            # 没有中文字体时用库内编号代替姓名
            if font is None and any(ord(ch) > 127 for ch in str(label)):
                label = f"#{target.get('entry_id')} {target.get('similarity')}"
        else:
            label = f"{target['class_name']} {target['confidence']}"
        _label(draw, target["bbox"][0], target["bbox"][1], label, font, color)

    for face in faces:
        color = {"matched": "#00bb66", "unknown": "#ff9800"}.get(face["status"], "#9aa0a6")
        draw.rectangle(face["bbox"], outline=color, width=2)
        for x, y in face.get("landmarks") or []:
            draw.ellipse((x-1.5, y-1.5, x+1.5, y+1.5), fill="#ff4081")

    if poses:
        pose_support.draw_pil(canvas, [{"bbox": pose["bbox"], "keypoints": pose["keypoints"],
                                        "behavior": pose.get("behavior")} for pose in poses])
    return canvas


# ---------------------------------------------------------------- 主流程

def aggregate_image(src: Path, *, weights: str | None = None, conf: float | None = None,
                    classes: list[str] | None = None, imgsz: int = 640,
                    face: bool = True, pose: bool = True,
                    threshold: float = face_recognition.DEFAULT_MATCH_THRESHOLD,
                    face_confidence: float = face_recognition.DETECT_CONFIDENCE,
                    min_confidence: float = face_recognition.DEFAULT_QUALITY_CONFIDENCE,
                    min_size: int = face_recognition.DEFAULT_MIN_FACE_PIXELS,
                    min_sharpness: float = face_recognition.DEFAULT_MIN_SHARPNESS,
                    pose_weights: str = POSE_WEIGHTS, pose_confidence: float = DEFAULT_POSE_CONFIDENCE,
                    bind: str = DEFAULT_BIND_STRATEGY, bind_iou: float = DEFAULT_BIND_IOU,
                    save: bool = True) -> dict:
    """一次跑完 目标检测 → 人脸识别 → 姿势检测绑定，返回结构化结果 + 带标注的结果图。"""
    started = time.perf_counter()
    weights = weights or vision.DEFAULT_WEIGHTS
    if bind not in BIND_STRATEGIES:
        raise ValueError(f"未知的绑定策略：{bind}（可选：{'、'.join(BIND_STRATEGIES)}）")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("识别阈值必须在 0 到 1 之间")
    if not math.isfinite(bind_iou) or not 0 <= bind_iou <= 1:
        raise ValueError("绑定 IoU 阈值必须在 0 到 1 之间")

    image = read_image(src)
    height, width = image.shape[:2]
    conf_used = vision.resolve_confidence(weights, conf)
    steps: dict[str, dict] = {}

    # ---------- 1) 目标检测 ----------
    mark = time.perf_counter()
    detection = detect_targets(src, weights, conf_used, classes, imgsz)
    targets = detection["detections"]
    steps["detect"] = {"enabled": True, "weight": detection["weight"], "confidence": conf_used,
                       "imgsz": detection["imgsz"], "mode": detection["mode"],
                       "classes_used": detection["classes_used"], "count": len(targets),
                       "elapsed_seconds": round(time.perf_counter()-mark, 3)}

    # ---------- 2) 人脸识别：姓名贴到最贴合的目标 ----------
    faces: list[dict] = []
    if not face:
        steps["face"] = {"enabled": False, "reason": "未勾选人脸识别"}
    else:
        if not face_recognition.list_entries():
            raise ValueError("人脸库为空：请先在「人脸检测」页导入学生花名册或添加人脸")
        mark = time.perf_counter()
        recognized = face_recognition.recognize_faces(
            image, threshold=threshold, detect_confidence=face_confidence,
            min_confidence=min_confidence, min_size=min_size, min_sharpness=min_sharpness)
        faces = recognized["faces"]
        bindings = bind_faces(faces, targets)
        named = 0
        for record in faces:
            binding = bindings.get(record["face_id"])
            record["bind"] = binding
            record["target_index"] = binding["target_index"] if binding else None
            target = targets[binding["target_index"]] if binding else None
            # 同一目标被多张脸命中时，先到先得（faces 已按人脸面积降序，大脸更可靠）
            if target is None or record["status"] != "matched" or "name" in target:
                continue
            target.update({"name": record["match"]["name"], "entry_id": record["match"]["entry_id"],
                           "similarity": record["match"]["similarity"],
                           "photo_url": record["match"].get("photo_url"), "face_id": record["face_id"]})
            named += 1
        steps["face"] = {"enabled": True, "threshold": threshold, "detect_confidence": face_confidence,
                         "min_size": min_size, "min_sharpness": min_sharpness,
                         "library_count": len(face_recognition.list_entries()),
                         "detected": recognized["detected"], "face_count": recognized["face_count"],
                         "matched": recognized["matched"], "unknown": recognized["unknown"],
                         "rejected": recognized["rejected"], "reassigned": recognized["reassigned"],
                         "truncated": recognized["truncated"], "named_targets": named,
                         "elapsed_seconds": round(time.perf_counter()-mark, 3)}

    # ---------- 3) 姿势检测 + 绑定 ----------
    poses: list[dict] = []
    if not pose:
        steps["pose"] = {"enabled": False, "reason": "未勾选姿势检测"}
    else:
        mark = time.perf_counter()
        reuse = bool(detection["keypoints"]) and len(detection["keypoints"]) == len(targets)
        if reuse:
            poses = [{"person_index": index, "bbox": target["bbox"], "confidence": target["confidence"],
                      "keypoints": points, "behavior": pose_behaviors.classify(points, int(width), int(height)),
                      "source": "detect_weight"}
                     for index, (target, points) in enumerate(zip(targets, detection["keypoints"]), 1)]
            pose_weight, pose_conf = detection["weight"], conf_used
        else:
            poses = detect_poses(src, pose_weights, pose_confidence, imgsz)
            pose_weight, pose_conf = Path(pose_weights).name, pose_confidence
        pose_bindings = bind_poses(poses, targets, bind, bind_iou)
        for index, record in enumerate(poses):
            binding = pose_bindings.get(index)
            record["bind"] = binding
            record["target_index"] = binding["target_index"] if binding else None
            if binding:
                targets[binding["target_index"]].setdefault("poses", []).append(
                    {"keypoints": record["keypoints"], "behavior": record["behavior"],
                     "confidence": record["confidence"], "bind": binding})
        steps["pose"] = {"enabled": True, "weight": pose_weight, "confidence": pose_conf,
                         "count": len(poses), "bound": len(pose_bindings),
                         "unbound": len(poses)-len(pose_bindings), "strategy": bind,
                         "iou_threshold": bind_iou,
                         "keypoints_source": "detect_weight" if reuse else "pose_model",
                         "elapsed_seconds": round(time.perf_counter()-mark, 3)}

    # ---------- 结果图 ----------
    out_name = ""
    if save:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = src.suffix.lower() if src.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp") else ".jpg"
        out_name = f"{src.stem}_agg{suffix}"
        render(image, targets, faces, poses).save(RESULT_DIR/out_name)

    return {
        "source": src.name,
        "result": out_name,
        "result_url": f"/api/aggregate/results/{quote(out_name)}" if out_name else "",
        "image_width": int(width), "image_height": int(height),
        "weight": Path(weights).name, "mode": detection["mode"], "imgsz": detection["imgsz"],
        "count": len(targets), "targets": targets, "faces": faces, "poses": poses,
        "steps": steps,
        "bind": {"strategy": bind, "rule": BIND_STRATEGIES[bind], "iou_threshold": bind_iou,
                 "face_tolerance": FACE_BIND_TOLERANCE},
        "unbound_faces": [f["face_id"] for f in faces if f.get("target_index") is None],
        "unbound_poses": [p["person_index"] for p in poses if p.get("target_index") is None],
        "summary": {
            "targets": len(targets), "named_targets": sum(1 for t in targets if t.get("name")),
            "faces": len(faces), "bound_faces": sum(1 for f in faces if f.get("target_index") is not None),
            "matched_faces": sum(1 for f in faces if f["status"] == "matched"),
            "unknown_faces": sum(1 for f in faces if f["status"] == "unknown"),
            "rejected_faces": sum(1 for f in faces if f["status"] == "rejected"),
            "poses": len(poses), "bound_poses": sum(1 for p in poses if p.get("target_index") is not None),
        },
        "elapsed_seconds": round(time.perf_counter()-started, 3),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
