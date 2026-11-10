"""聚合操作：单图 目标检测 → 人脸识别（姓名贴到人头）→ 姿势检测与人头绑定。

三步互相独立、后两步可分别开关：
1. 目标检测：固定复用「图片推理 → 目标检测」那套人头检测（medium.pt 0.45 为基准、
   RT-DETRv4 0.25 补齐；合并 IoU ≤ 0.1、自定义 IoU 生效置信度 > 0.35 在这里锁死，
   前端不暴露这些开关）；
2. 人脸识别：整图 SCRFD 检出所有人脸 → 质量筛选 → ArcFace 提特征 → 1:N 比对，
   并把识别到的**姓名贴到最贴合的人头框**上（人脸中心落在哪个人头框内就归给谁，
   多个框都包含时取面积最小的；都不包含时退化为最近人头框中心，超出容差就不归属）；
3. 姿势检测：Pose 权重出 17 点骨架，按选定策略（人头中心包含 / 最近中心 / IoU）与人头绑定。

结果图由后端画好（人头框 + 姓名标签 + 人脸框 + 骨架），前端直接展示即可。
"""
import math
import copy
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import face_recognition
import head_fusion
import vision
import aggregate_actions
from action_recognition import jobs as action_jobs, vlm
from project_paths import OUTPUT_DIR

# 结果图目录：独立子目录，避免和「图片推理」的 data/results 混在一起
RESULT_DIR = OUTPUT_DIR / "aggregate"

POSE_WEIGHTS = "yolov8m-pose.pt"
DEFAULT_POSE_CONFIDENCE = 0.6
POSE_IMGSZ = 640
POSE_HEAD_DOWN_R = 0.05
POSE_HEAD_UP_R = 0.20
POSE_OBJECT_CONFIDENCE = 0.03

# 目标检测：与「图片推理 → 目标检测」完全一致，以 medium.pt 为基准（False）
RTDETR_BASE = False
MERGE_IOU = 0.1
MERGE_CONFIDENCE = 0.35

# 绑定规则（固定，不对外暴露）：只要求 IoU > 0，链式贪心 + 一一对应
MIN_BIND_IOU = 0.0
BIND_RULE = ("链式贪心 + 一一对应：只保留 IoU > 0 的配对，按 IoU 降序占位（IoU 相同时中心距离近的先占），"
             "一个人头最多绑一个姿势、一个姿势最多绑一个人头")
FACE_BIND_RULE = ("链式贪心 + 一一对应：只保留 IoU > 0 的配对，按 IoU 降序占位（IoU 相同时中心距离近的先占），"
                  "一个人头最多贴一个姓名、一张脸最多贴一个人头")

# 人脸识别参数全部锁死（界面上只有一个「人脸识别」勾选框，不开放任何阈值输入）：
# 人脸置信度 0.5（SCRFD 检出下限）、识别阈值 0.3（余弦相似度下限）、
# 质量门槛沿用 face_recognition 查询端默认值（0.5 / 24px / 50）。
FACE_CONFIDENCE = 0.5
MATCH_THRESHOLD = 0.3
MIN_QUALITY_CONFIDENCE = face_recognition.DEFAULT_QUALITY_CONFIDENCE
MIN_FACE_PIXELS = face_recognition.DEFAULT_MIN_FACE_PIXELS
MIN_FACE_SHARPNESS = face_recognition.DEFAULT_MIN_SHARPNESS

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


def _distance(a, b) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


# ---------------------------------------------------------------- 绑定

def match_one_to_one(items: list[tuple], targets: list[dict]) -> dict:
    """链式贪心 + 一一对应：把 items（键 → 框）配到 targets 上。

    - 只保留 IoU > MIN_BIND_IOU 的配对：IoU = 0 就是根本不相交，不硬塞；
    - 所有配对按 IoU 降序处理，IoU 相同时中心距离更近的先占；
    - 一个目标只能被一个 item 占用，一个 item 也只能占一个目标；
      被抢走的 item 不会改去占已被人占的目标，就此判为未绑定（宁可漏绑也不乱绑）。

    返回 {item_key: {target_index, iou, distance}}。
    """
    if not targets:
        return {}
    pairs: list[tuple[float, float, int, object, int]] = []
    for order, (key, box) in enumerate(items):
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            continue
        for index, target in enumerate(targets):
            value = iou(box, target["bbox"])
            if value <= MIN_BIND_IOU:
                continue
            pairs.append((value, _distance(_center(box), _center(target["bbox"])), order, key, index))
    # (-IoU, 中心距离)：并列时距离近的先占；仍完全并列则按传入顺序（sort 稳定）
    pairs.sort(key=lambda item: (-item[0], item[1]))
    taken: set[int] = set()
    matched: dict = {}
    for value, distance, _order, key, index in pairs:
        if key in matched or index in taken:
            continue
        taken.add(index)
        matched[key] = {"target_index": index, "iou": round(value, 4), "distance": round(distance, 2)}
    return matched


def bind_faces(faces: list[dict], targets: list[dict]) -> dict:
    """人脸 → 人头：一一对应，识别到的姓名就贴在这个人头上。

    人脸框落在人头框里时 IoU = 人脸面积 / 人头面积 > 0，所以“IoU > 0”既筛掉了
    完全不沾边的脸，又允许多人头场景下按 IoU（=贴合程度）排序后再一一分配。
    """
    return match_one_to_one([(face["face_id"], face.get("bbox") or []) for face in faces], targets)


def bind_poses(poses: list[dict], targets: list[dict]) -> dict:
    """姿势 → 人头：一一对应，链式贪心取 IoU 最大的。

    人头框远小于人体框，但人头在人体框内 → IoU = 人头面积 / 人体框面积 > 0；
    人头框与人体框不相交的姿势会判为未绑定，而不是硬塞给某个人头。
    """
    return match_one_to_one([(index, pose.get("bbox") or []) for index, pose in enumerate(poses)], targets)


# ---------------------------------------------------------------- 三步推理

def read_image(src: Path) -> np.ndarray:
    """读原图（中文路径安全）。"""
    image = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{src.name}")
    return image


def detect_targets(src: Path) -> dict:
    """目标检测：固定复用「图片推理 → 目标检测」的人头检测。

    medium.pt 0.45 为基准 + RT-DETRv4 0.25 补齐；合并 IoU ≤ 0.1、自定义 IoU 生效
    置信度 > 0.35 全部锁死在这里，前端不暴露——避免两套目标检测参数各自漂移。
    """
    result = head_fusion.detect_image(src, rtdetr_base=RTDETR_BASE,
                                      merge_iou=MERGE_IOU,
                                      merge_confidence=MERGE_CONFIDENCE)
    detections = [{"class_id": int(d.get("class_id", 0)), "class_name": d["class_name"],
                   "confidence": round(float(d["confidence"]), 4),
                   "bbox": [round(float(v), 2) for v in d["bbox"]],
                   "source_model": d.get("source_model")}
                  for d in result["detections"]]
    fusion = result.get("fusion") or {}
    rule = (f"medium.pt {head_fusion.BASE_CONFIDENCE} 为基准 + "
            f"{fusion.get('incoming_weight', '')} {head_fusion.HEAD_CONFIDENCE} 补齐；"
            f"合并 IoU ≤ {fusion.get('merge_iou', MERGE_IOU)}、"
            f"自定义 IoU 生效置信度 > {fusion.get('merge_confidence', MERGE_CONFIDENCE)}")
    return {"detections": detections, "mode": result["mode"], "imgsz": result["imgsz"],
            "classes_used": result.get("classes_used", []), "weight": result.get("weight"),
            "fusion": fusion, "rule": rule}


def detect_best_targets(src: Path, image, progress=None) -> dict:
    original = detect_targets(src)
    aggregate_actions.emit(progress, 'detect', '原图检测完成，正在检测水平镜像', 1, 2)
    with tempfile.TemporaryDirectory(prefix='aggregate-mirror-') as folder:
        mirrored_source = Path(folder) / 'mirror.png'
        encoded, pixels = cv2.imencode('.png', cv2.flip(image, 1))
        if not encoded:
            raise ValueError('镜像图片生成失败')
        pixels.tofile(str(mirrored_source))
        mirrored = detect_targets(mirrored_source)
    original_count = len(original['detections'])
    mirror_count = len(mirrored['detections'])
    use_mirror = mirror_count > original_count
    selected = copy.deepcopy(mirrored if use_mirror else original)
    if use_mirror:
        width = image.shape[1]
        for target in selected['detections']:
            x1, y1, x2, y2 = target['bbox']
            target['bbox'] = [round(width - x2, 2), y1, round(width - x1, 2), y2]
    selected['mirror_selection'] = {
        'original_count': original_count, 'mirror_count': mirror_count,
        'selected': 'mirror' if use_mirror else 'original', 'coordinates': 'original',
    }
    selected['rule'] += f'；原图 {original_count} 个、镜像 {mirror_count} 个，采用{"镜像" if use_mirror else "原图"}检测结果，数量相同优先原图'
    return selected


def detect_poses(src: Path) -> dict:
    """固定复用图片 Pose 流程：排除无法判断者，再执行全图物体检测与关联。"""
    result = vision.detect_image(
        src, weights=POSE_WEIGHTS, conf=DEFAULT_POSE_CONFIDENCE, imgsz=POSE_IMGSZ,
        save_result=False, head_down_r_threshold=POSE_HEAD_DOWN_R,
        head_up_r_threshold=POSE_HEAD_UP_R, pose_roi=True,
        roi_confidence=POSE_OBJECT_CONFIDENCE)
    if result["mode"] != "pose":
        raise ValueError(f"权重 {POSE_WEIGHTS} 没有返回骨架")
    return result


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


def render(image: np.ndarray, targets: list[dict]) -> Image.Image:
    """把结果画到一张图上：**只画 head 框**，姓名 / 置信度 / 姿势状态都挂在同一个框的标签上。

    人脸框与骨架不单独画——它们只是归属依据，最终展示的是“这个人头是谁、什么姿态”。
    """
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font = _font(18)

    for target in targets:
        name = target.get("name")
        color = "#00bb66" if name else "#2f6fed"
        draw.rectangle(target["bbox"], outline=color, width=3)
        if name:
            # 没有中文字体时用库内编号代替姓名
            shown = name if font is not None or all(ord(ch) < 128 for ch in str(name)) else f"#{target.get('entry_id')}"
            label = f"{shown} {target.get('similarity')}"
        else:
            label = f"{target['class_name']} {target['confidence']}"
        states = list(dict.fromkeys(
            (pose.get("behavior") or {}).get("state") for pose in target.get("poses", [])))
        states = [state for state in states if state]
        action = target.get('action') or {}
        if action.get('state'):
            states = [action['state']]
        elif action.get('status') == 'failed':
            states = ['动作分析失败']
        if states:
            label += " · " + "、".join(states)
        _label(draw, target["bbox"][0], target["bbox"][1], label, font, color)
    return canvas


# ---------------------------------------------------------------- 主流程

def aggregate_image(src: Path, *, face: bool = True, pose: bool = True, save: bool = True,
                    multimodal: bool = False, progress: aggregate_actions.Progress | None = None,
                    mirror_selection: bool = True) -> dict:
    aggregate_actions.emit(progress, 'prepare', '准备图片与模型资源')
    if multimodal:
        with action_jobs.mutation():
            ready = vlm.status(vlm.DEFAULT_MODEL, '4bit')
            if not ready['available']:
                raise ValueError(ready['message'])
            vlm.unload()
            return _aggregate_image(src, face=face, pose=pose, save=save, multimodal=True, progress=progress, mirror_selection=mirror_selection)
    return _aggregate_image(src, face=face, pose=pose, save=save, progress=progress, mirror_selection=mirror_selection)


def _aggregate_image(src: Path, *, face: bool = True, pose: bool = True, save: bool = True,
                     multimodal: bool = False, progress: aggregate_actions.Progress | None = None,
                    mirror_selection: bool = True) -> dict:
    """一次跑完 目标检测 → 人脸识别 → 姿势检测绑定，返回结构化结果 + 带标注的结果图。

    勾选「多模态姿态识别」时，固定采用先全体 Head、再全体扩展 ROI 的策略一：
    目标检测就是「图片推理 → 目标检测」那一套；人脸识别固定用 FACE_CONFIDENCE /
    MATCH_THRESHOLD（生效值回报在 steps.face 里）；绑定规则固定（链式贪心 + 一一对应）。
    """
    started = time.perf_counter()

    image = read_image(src)
    height, width = image.shape[:2]
    steps: dict[str, dict] = {}
    aggregate_actions.emit(progress, 'prepare', '图片准备完成', 1, 1)

    # ---------- 1) 目标检测（固定：图片推理的目标检测）----------
    mark = time.perf_counter()
    aggregate_actions.emit(progress, 'detect', '正在执行目标检测')
    detection = detect_best_targets(src, image, progress) if mirror_selection else detect_targets(src)
    targets = detection["detections"]
    aggregate_actions.emit(progress, 'detect', f'目标检测完成：{len(targets)} 个人头', 1, 1)
    steps["detect"] = {"enabled": True, "mode": detection["mode"], "weight": detection["weight"],
                       "imgsz": detection["imgsz"], "classes_used": detection["classes_used"],
                       "count": len(targets), "rule": detection["rule"], "fusion": detection["fusion"],
                       "mirror_selection": detection.get("mirror_selection"),
                       "elapsed_seconds": round(time.perf_counter()-mark, 3)}

    # ---------- 2) 人脸识别：姓名贴到最贴合的人头 ----------
    faces: list[dict] = []
    if not face:
        steps["face"] = {"enabled": False, "reason": "未勾选人脸识别"}
    else:
        aggregate_actions.emit(progress, 'face', '正在执行人脸识别')
        if not face_recognition.list_entries():
            raise ValueError("人脸库为空：请先在「人脸检测」页导入学生花名册或添加人脸")
        mark = time.perf_counter()
        recognized = face_recognition.recognize_faces(
            image, threshold=MATCH_THRESHOLD, detect_confidence=FACE_CONFIDENCE,
            min_confidence=MIN_QUALITY_CONFIDENCE, min_size=MIN_FACE_PIXELS,
            min_sharpness=MIN_FACE_SHARPNESS)
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
        steps["face"] = {"enabled": True, "threshold": MATCH_THRESHOLD,
                         "detect_confidence": FACE_CONFIDENCE,
                         "min_size": MIN_FACE_PIXELS, "min_sharpness": MIN_FACE_SHARPNESS,
                         "library_count": len(face_recognition.list_entries()),
                         "detected": recognized["detected"], "face_count": recognized["face_count"],
                         "matched": recognized["matched"], "unknown": recognized["unknown"],
                         "rejected": recognized["rejected"], "reassigned": recognized["reassigned"],
                         "truncated": recognized["truncated"], "named_targets": named,
                         "elapsed_seconds": round(time.perf_counter()-mark, 3)}
        aggregate_actions.emit(progress, 'face', '人脸识别完成', 1, 1)

    # ---------- 3) 姿势检测 + 绑定 ----------
    poses: list[dict] = []
    steps['multimodal'] = {'enabled': False, 'reason': '未勾选多模态姿态识别'}
    if not pose and not multimodal:
        steps["pose"] = {"enabled": False, "reason": "未勾选姿势检测"}
    elif multimodal:
        steps['pose'] = {'enabled': False, 'reason': '动作判断改用 VLM，不运行 Pose 规则'}
        directory = RESULT_DIR / 'vlm' / uuid.uuid4().hex if save else None
        steps['multimodal'] = aggregate_actions.analyze_heads(image, targets, directory, progress)
    else:
        mark = time.perf_counter()
        aggregate_actions.emit(progress, 'pose', '正在执行 Pose 姿势检测与绑定')
        pose_result = detect_poses(src)
        poses = [{"person_index": index, "bbox": item["bbox"],
                  "confidence": item["confidence"], "keypoints": item["keypoints"],
                  "behavior": item["behavior"]}
                 for index, item in enumerate(pose_result["detections"], 1)]
        pose_bindings = bind_poses(poses, targets)
        for index, record in enumerate(poses):
            binding = pose_bindings.get(index)
            record["bind"] = binding
            record["target_index"] = binding["target_index"] if binding else None
            if binding:
                targets[binding["target_index"]].setdefault("poses", []).append(
                    {"keypoints": record["keypoints"], "behavior": record["behavior"],
                     "confidence": record["confidence"], "bind": binding})
        steps["pose"] = {"enabled": True, "weight": POSE_WEIGHTS, "confidence": DEFAULT_POSE_CONFIDENCE,
                         "imgsz": POSE_IMGSZ, "head_down_r_threshold": POSE_HEAD_DOWN_R,
                         "head_up_r_threshold": POSE_HEAD_UP_R, "pose_roi": True,
                         "roi_confidence": POSE_OBJECT_CONFIDENCE,
                         "excluded_unknown": pose_result["excluded_unknown"],
                         "count": len(poses), "bound": len(pose_bindings),
                         "unbound": len(poses)-len(pose_bindings), "rule": BIND_RULE,
                         "min_iou": MIN_BIND_IOU,
                         "elapsed_seconds": round(time.perf_counter()-mark, 3)}
        aggregate_actions.emit(progress, 'pose', 'Pose 姿势检测与绑定完成', 1, 1)

    # ---------- 结果图（只画 head 框）----------
    out_name = ""
    aggregate_actions.emit(progress, 'render', '正在生成聚合结果')
    if save:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = src.suffix.lower() if src.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp") else ".jpg"
        out_name = f"{src.stem}_agg{suffix}"
        render(image, targets).save(RESULT_DIR/out_name)

    aggregate_actions.emit(progress, 'render', '聚合结果生成完成', 1, 1)

    return {
        "source": src.name,
        "result": out_name,
        "result_url": f"/api/aggregate/results/{quote(out_name)}" if out_name else "",
        "image_width": int(width), "image_height": int(height),
        "mode": detection["mode"], "imgsz": detection["imgsz"],
        "count": len(targets), "targets": targets, "faces": faces, "poses": poses,
        "steps": steps,
        "bind": {"rule": BIND_RULE, "face_rule": FACE_BIND_RULE, "min_iou": MIN_BIND_IOU},
        "unbound_faces": [f["face_id"] for f in faces if f.get("target_index") is None],
        "unbound_poses": [p["person_index"] for p in poses if p.get("target_index") is None],
        "summary": {
            "targets": len(targets), "named_targets": sum(1 for t in targets if t.get("name")),
            "faces": len(faces), "bound_faces": sum(1 for f in faces if f.get("target_index") is not None),
            "matched_faces": sum(1 for f in faces if f["status"] == "matched"),
            "unknown_faces": sum(1 for f in faces if f["status"] == "unknown"),
            "rejected_faces": sum(1 for f in faces if f["status"] == "rejected"),
            "poses": len(poses), "bound_poses": sum(1 for p in poses if p.get("target_index") is not None),
            'vlm_actions': sum((t.get('action') or {}).get('status') == 'done' for t in targets),
            'failed_actions': sum((t.get('action') or {}).get('status') == 'failed' for t in targets),
        },
        "elapsed_seconds": round(time.perf_counter()-started, 3),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
