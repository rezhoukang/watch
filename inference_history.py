"""Immutable image inference records and directed, base-first result merges."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


HISTORY_DIR = Path(__file__).resolve().parent / "data" / "inferences"


def record_dir(record_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", record_id):
        raise ValueError("非法记录 ID")
    return HISTORY_DIR / record_id


def begin_record(src: Path) -> tuple[str, Path]:
    record_id = uuid.uuid4().hex
    directory = record_dir(record_id) / "source"
    directory.mkdir(parents=True)
    snapshot = directory / src.name
    shutil.copyfile(src, snapshot)
    return record_id, snapshot


def read_record(record_id: str) -> dict:
    return json.loads((record_dir(record_id) / "record.json").read_text(encoding="utf-8"))


def delete_record(record_id: str) -> None:
    directory = record_dir(record_id)
    # Validate the absolute target immediately before recursive deletion. Never
    # follow a record symlink or remove the history root itself.
    if directory.is_symlink() or directory.resolve() != HISTORY_DIR.resolve() / record_id:
        raise ValueError("非法推理记录目录")
    if not (directory / "record.json").is_file():
        raise FileNotFoundError(record_id)
    shutil.rmtree(directory)


def source_path(record: dict) -> Path:
    name = record["source"]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("非法图片文件名")
    return record_dir(record["record_id"]) / "source" / name


def _fingerprint(src: Path) -> str:
    from PIL import Image

    with Image.open(src) as opened:
        image = opened.convert("RGBA")
    digest = hashlib.sha256(str(image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def complete_record(record_id: str, src: Path, result: dict, parameters: dict, *,
                    kind: str = "inference", merge: dict | None = None) -> dict:
    from PIL import Image, ImageDraw

    directory = record_dir(record_id)
    with Image.open(src) as opened:
        image = opened.convert("RGBA")
    record = copy.deepcopy(result)
    record.update({
        "record_id": record_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "source": src.name,
        "input_sha256": _fingerprint(src),
        "parameters": parameters,
        "image_width": image.width,
        "image_height": image.height,
        "source_url": f"/api/inferences/{record_id}/image/source",
        "result": "result.png",
        "result_url": f"/api/inferences/{record_id}/image/result",
        "record_url": f"/api/inferences/{record_id}",
    })
    if merge:
        record["merge"] = merge
    original_alpha = image.getchannel("A")
    draw = ImageDraw.Draw(image)
    for det in record["detections"]:
        incoming = (merge and det.get("origin_record_id") == merge["incoming_id"]) or (
            record.get("fusion") and det.get("source_model") == record["fusion"]["incoming_weight"])
        color = "#ff9900" if incoming else "#00bb66"
        draw.rectangle(det["bbox"], outline=color, width=2)
    image.putalpha(original_alpha)
    if any(det.get("keypoints") for det in record["detections"]):
        import pose_support

        record["pose"] = pose_support.metadata()
        pose_support.draw_pil(image, record["detections"])
        image.putalpha(original_alpha)
    image.save(directory / "result.png")
    # Publish metadata last: incomplete/failed inferences never appear in the list.
    temporary = directory / "record.json.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "record.json")
    return record


def list_records() -> list[dict]:
    fields = ("record_id", "created_at", "kind", "source", "input_sha256", "weight", "mode",
              "model_name", "count", "head_count", "person_count", "head_count_only", "imgsz",
              "source_url", "result_url", "parameters", "merge", "fusion", "elapsed_seconds")
    records = []
    for path in HISTORY_DIR.glob("*/record.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        records.append({k: record[k] for k in fields if k in record})
    return sorted(records, key=lambda r: (r["created_at"], r["record_id"]), reverse=True)


def validate_merge_iou(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("合并 IoU 阈值必须是 0 到 1 之间的数字")
    return float(value)


def validate_merge_confidence(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("自定义 IoU 的置信度门槛必须是 0 到 1 之间的数字")
    return float(value)


def box_iou(a: list, b: list) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def overlaps(a: list, b: list, merge_iou: float = 0.0) -> bool:
    return box_iou(a, b) > merge_iou


def merge_detections(base: list[dict], incoming: list[dict], merge_iou: float = 0.0, merge_confidence: float = 0.0) -> tuple[list[dict], int, int]:
    """Keep every baseline box; reject incoming boxes only above the IoU limit."""
    merge_iou = validate_merge_iou(merge_iou)
    merge_confidence = validate_merge_confidence(merge_confidence)
    added = []
    for det in incoming:
        threshold = merge_iou if det.get("confidence", 0) > merge_confidence else 0.0
        if not any(overlaps(det["bbox"], kept["bbox"], threshold) for kept in base):
            added.append(det)
    return copy.deepcopy(base + added), len(added), len(incoming) - len(added)


def merge_records(base_id: str, incoming_id: str, merge_iou: float = 0.0, merge_confidence: float = 0.0) -> dict:
    merge_iou = validate_merge_iou(merge_iou)
    merge_confidence = validate_merge_confidence(merge_confidence)
    if base_id == incoming_id:
        raise ValueError("请选择两条不同的推理记录")
    base, incoming = read_record(base_id), read_record(incoming_id)
    if base["input_sha256"] != incoming["input_sha256"]:
        raise ValueError("只能合并同一张原图的推理记录（图片内容及尺寸须一致）")
    base_dets = copy.deepcopy(base["detections"])
    for det in base_dets:
        det["origin_record_id"] = base_id
    incoming_dets = copy.deepcopy(incoming["detections"])
    for det in incoming_dets:
        det["origin_record_id"] = incoming_id
    detections, added_count, discarded = merge_detections(base_dets, incoming_dets, merge_iou, merge_confidence)
    summary = {}
    for det in detections:
        name = det["class_name"]
        summary[name] = summary.get(name, 0) + 1
    classes = sorted(set(base.get("classes_used", [])) | set(incoming.get("classes_used", [])))
    result = {
        "mode": "merged", "model_name": "合并推理结果", "weight": base.get("weight", ""),
        "imgsz": base.get("imgsz", 640), "classes_used": classes, "count": len(detections),
        "detections": detections, "summary": summary, "head_count": summary.get("head", 0),
        "head_count_only": classes == ["head"], "elapsed_seconds": 0,
    }
    if result["head_count_only"]:
        result["person_count"] = result["head_count"]
    record_id, snapshot = begin_record(source_path(base))
    return complete_record(record_id, snapshot, result, {"base": base["parameters"], "incoming": incoming["parameters"], "merge_iou": merge_iou, "merge_confidence": merge_confidence},
                           kind="merge", merge={"base_id": base_id, "incoming_id": incoming_id,
                                                "discarded_overlaps": discarded, "added_count": added_count,
                                                "merge_iou": merge_iou,
                                                "merge_confidence": merge_confidence,
                                                "rule": "confidence_gated_iou_keep_base"})
