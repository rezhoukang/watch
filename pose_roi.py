"""Bind full-image objects to head-down people by maximum ROI intersection."""
import math
import threading
from typing import Any

from project_paths import MODELS_DIR

WEIGHTS = "yolov8m-worldv2.pt"
PROMPTS = ["cell phone", "book"]
DEFAULT_CONFIDENCE = 0.03
RULE = {"version": "full_image_objects_v4", "weights": WEIGHTS, "prompts": PROMPTS,
        "roi_source": "pose_bbox",
        "association": "maximum_positive_intersection_area", "detection_scope": "full_image",
        "confidence": DEFAULT_CONFIDENCE, "imgsz": 640}
_lock = threading.RLock()
_model: Any = None


def roi_polygon(box: list, width: int, height: int):
    import numpy as np

    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLOWorld

        path = MODELS_DIR / WEIGHTS
        if not path.is_file():
            raise FileNotFoundError(f"物体检测权重不存在：{WEIGHTS}")
        # A private instance keeps normal World prompt changes out of this pipeline.
        model = YOLOWorld(str(path))
        model.set_classes(PROMPTS)
        _model = model
    return _model


def clear_cache() -> int:
    global _model
    with _lock:
        count = int(_model is not None)
        _model = None
        return count


def _intersection(box: list, polygon) -> float:
    import cv2
    import numpy as np

    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1 or not all(math.isfinite(v) for v in box):
        return 0.0
    rectangle = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    area, _ = cv2.intersectConvexConvex(polygon, rectangle)
    return float(area)


def _iou(a, b) -> float:
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union > 0 else 0.0


def refine(frame, detections: list[dict], enabled: bool = True, confidence: float = DEFAULT_CONFIDENCE,
           device: int | str = "cpu") -> None:
    """Mutate only head-down states; preserve the head evidence and person count."""
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("物体置信度必须在 0 到 1 之间")
    height, width = frame.shape[:2]
    candidates = []
    for index, detection in enumerate(detections):
        behavior = detection.get("behavior", {})
        if behavior.get("state") != "低头":
            continue
        evidence = {"status": "disabled" if not enabled else "invalid_pose_bbox",
                    "rule": {**RULE, "confidence": confidence}, "objects": []}
        behavior["roi"] = evidence
        if not enabled:
            continue
        polygon = roi_polygon(detection.get("bbox", []), width, height)
        if polygon is None:
            continue
        evidence.update({"polygon": polygon.tolist(), "status": "pending"})
        candidates.append((index, polygon))
    if not candidates:
        return
    try:
        with _lock:
            model = _get_model()
            predictions = model.predict(source=frame, conf=confidence, imgsz=640, device=device, verbose=False)
        if len(predictions) != 1:
            raise RuntimeError("全图物体检测应返回一张图片的结果")
    except Exception as exc:
        # No reliable object evidence means retain head-down, rather than guess.
        for index, *_ in candidates:
            detections[index]["behavior"]["roi"].update({"status": "unavailable", "message": str(exc)})
        return
    objects = []
    for index, _ in candidates:
        detections[index]["behavior"]["roi"]["status"] = "no_reliable_object"
    for box in predictions[0].boxes:
        cls, score = int(box.cls[0]), float(box.conf[0])
        if cls not in (0, 1) or not math.isfinite(score) or score < confidence:
            continue
        a, b, c, d = (float(v) for v in box.xyxy[0].tolist())
        if not all(math.isfinite(v) for v in (a, b, c, d)):
            continue
        bbox = [max(0, a), max(0, b), min(width, c), min(height, d)]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        objects.append({"class_name": PROMPTS[cls], "kind": "phone" if cls == 0 else "reading",
                        "confidence": round(score, 4), "bbox": bbox})
    # Deduplicate predictions of the same physical item before binding.
    accepted = []
    for obj in sorted(objects, key=lambda v: -v["confidence"]):
        if any(obj["kind"] == prior["kind"] and _iou(obj["bbox"], prior["bbox"]) > .5 for prior in accepted):
            continue
        accepted.append(obj)
        bbox = obj["bbox"]
        matches = []
        for index, polygon in candidates:
            area = _intersection(bbox, polygon)
            if area <= 0:
                continue
            matches.append((area, -index))
        if not matches:
            continue
        area, neg_index = max(matches)
        owner = -neg_index
        detections[owner]["behavior"]["roi"]["objects"].append({**obj,
            "bbox": [round(v, 2) for v in bbox], "intersection_area": round(area, 4), "person_index": owner})
    for index, *_ in candidates:
        behavior = detections[index]["behavior"]
        evidence = behavior["roi"]
        if evidence["objects"]:
            evidence["status"] = "matched"
            behavior["state"] = "疑似看手机" if any(o["kind"] == "phone" for o in evidence["objects"]) else "疑似阅读"
