"""Merge YOLO medium and RT-DETRv4 with a selectable overlap baseline."""
import copy
from pathlib import Path

import rtdetrv4
import vision
from inference_history import merge_detections, validate_merge_iou, validate_merge_confidence

BASE_WEIGHTS = "medium.pt"
BASE_CONFIDENCE = 0.45
HEAD_CONFIDENCE = 0.25
IMAGE_SIZE = 640
DEFAULT_MERGE_IOU = 0.1
DEFAULT_MERGE_CONFIDENCE = 0.45


def parameters(rtdetr_base: bool = False, merge_iou: float = DEFAULT_MERGE_IOU, merge_confidence: float = DEFAULT_MERGE_CONFIDENCE) -> dict:
    config = {
        "head_detection": True, "imgsz": IMAGE_SIZE,
        "rtdetr_base": rtdetr_base,
        "merge_iou": validate_merge_iou(merge_iou),
        "merge_confidence": validate_merge_confidence(merge_confidence),
        "base": {"weights": BASE_WEIGHTS, "confidence": BASE_CONFIDENCE, "imgsz": IMAGE_SIZE},
        "incoming": {"weights": rtdetrv4.WEIGHTS_NAME, "confidence": HEAD_CONFIDENCE, "imgsz": IMAGE_SIZE},
    }
    if rtdetr_base:
        config["base"], config["incoming"] = config["incoming"], config["base"]
    return config


def detect_image(src: Path, rtdetr_base: bool = False, merge_iou: float = DEFAULT_MERGE_IOU, merge_confidence: float = DEFAULT_MERGE_CONFIDENCE) -> dict:
    base = vision.detect_image(src, weights=BASE_WEIGHTS, conf=BASE_CONFIDENCE,
                               imgsz=IMAGE_SIZE, save_result=False)
    incoming = vision.detect_image(src, weights=rtdetrv4.WEIGHTS_NAME, conf=HEAD_CONFIDENCE,
                                   imgsz=IMAGE_SIZE, save_result=False)
    return fuse_results(base, incoming, rtdetr_base, merge_iou, merge_confidence)


def fuse_results(base: dict, incoming: dict, rtdetr_base: bool = False, merge_iou: float = DEFAULT_MERGE_IOU, merge_confidence: float = DEFAULT_MERGE_CONFIDENCE) -> dict:
    base_dets, incoming_dets = copy.deepcopy(base["detections"]), copy.deepcopy(incoming["detections"])
    for det in base_dets:
        det["source_model"] = BASE_WEIGHTS
    for det in incoming_dets:
        det["source_model"] = rtdetrv4.WEIGHTS_NAME
    config = parameters(rtdetr_base, merge_iou, merge_confidence)
    if rtdetr_base:
        base, incoming = incoming, base
        base_dets, incoming_dets = incoming_dets, base_dets
    detections, added, discarded = merge_detections(base_dets, incoming_dets, merge_iou, merge_confidence)
    classes = list(dict.fromkeys(base.get("classes_used", []) + [d["class_name"] for d in detections]))
    summary = {}
    for det in detections:
        # Model-local numeric IDs may collide; class names retain their meaning.
        det["source_class_id"] = det["class_id"]
        det["class_id"] = classes.index(det["class_name"])
        summary[det["class_name"]] = summary.get(det["class_name"], 0) + 1
    result = {
        "mode": "head_fusion", "model_name": "head检测（RT-DETRv4 基准 + YOLO medium）" if rtdetr_base else "head检测（YOLO medium 基准 + RT-DETRv4）",
        "weight": config["base"]["weights"], "imgsz": IMAGE_SIZE, "classes_used": classes,
        "count": len(detections), "detections": detections, "summary": summary,
        "head_count": summary.get("head", 0), "head_count_only": classes == ["head"],
        "fusion": {"base_weight": config["base"]["weights"], "incoming_weight": config["incoming"]["weights"],
                   "rtdetr_base": rtdetr_base,
                   "merge_iou": merge_iou,
                   "merge_confidence": merge_confidence,
                   "base_count": len(base["detections"]), "incoming_count": len(incoming["detections"]),
                   "added_count": added, "discarded_overlaps": discarded,
                   "rule": "confidence_gated_iou_keep_base"},
    }
    if result["head_count_only"]:
        result["person_count"] = result["head_count"]
    return result


def detect_frame(frame, model=None, rtdetr_base: bool = False, merge_iou: float = DEFAULT_MERGE_IOU, merge_confidence: float = DEFAULT_MERGE_CONFIDENCE) -> dict:
    """Fuse both models on the exact same BGR frame without temporary image files."""
    from PIL import Image
    model = model if model is not None else vision.get_model(BASE_WEIGHTS)
    prediction = model.predict(source=frame, conf=BASE_CONFIDENCE, imgsz=IMAGE_SIZE,
                               device=vision.device_arg(), verbose=False)[0]
    names = prediction.names
    detections = []
    for box in prediction.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = (round(float(value), 2) for value in box.xyxy[0].tolist())
        detections.append({"class_id": class_id, "class_name": names[class_id],
                           "confidence": round(float(box.conf[0]), 4), "bbox": [x1, y1, x2, y2],
                           "bbox_cxcywh": [round((x1+x2)/2, 2), round((y1+y2)/2, 2), round(x2-x1, 2), round(y2-y1, 2)]})
    heads, _ = rtdetrv4.detect_pil(Image.fromarray(frame[:, :, ::-1]), HEAD_CONFIDENCE, IMAGE_SIZE)
    classes = [str(value) for _, value in sorted(names.items())] if isinstance(names, dict) else list(names)
    return fuse_results({"detections": detections, "classes_used": classes}, {"detections": heads}, rtdetr_base, merge_iou, merge_confidence)


def annotate_frame(frame, result: dict, render: str, line_width: int):
    import cv2
    for det in result["detections"]:
        x1, y1, x2, y2 = (int(value) for value in det["bbox"])
        color = (102, 187, 0) if det["source_model"] == result.get("fusion", {}).get("base_weight", BASE_WEIGHTS) else (0, 153, 255)
        if render == "dot":
            cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 3, color, -1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)
        cv2.putText(frame, f'{det["class_name"]} {det["confidence"]:.2f}', (x1, max(12, y1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame
