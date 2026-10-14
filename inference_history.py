"""Immutable image inference records and directed, base-first result merges."""
from __future__ import annotations

import copy
import hashlib
import json
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
        color = "#ff9900" if merge and det.get("origin_record_id") == merge["incoming_id"] else "#00bb66"
        draw.rectangle(det["bbox"], outline=color, width=2)
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
              "source_url", "result_url", "parameters", "merge", "elapsed_seconds")
    records = []
    for path in HISTORY_DIR.glob("*/record.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        records.append({k: record[k] for k in fields if k in record})
    return sorted(records, key=lambda r: (r["created_at"], r["record_id"]), reverse=True)


def overlaps(a: list, b: list) -> bool:
    # Any positive intersection area, including across different classes.
    # Touching edges alone have zero area and are retained.
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def merge_records(base_id: str, incoming_id: str) -> dict:
    if base_id == incoming_id:
        raise ValueError("请选择两条不同的推理记录")
    base, incoming = read_record(base_id), read_record(incoming_id)
    if base["input_sha256"] != incoming["input_sha256"]:
        raise ValueError("只能合并同一张原图的推理记录（图片内容及尺寸须一致）")
    base_dets = copy.deepcopy(base["detections"])
    for det in base_dets:
        det["origin_record_id"] = base_id
    added, discarded = [], 0
    for det in incoming["detections"]:
        if any(overlaps(det["bbox"], kept["bbox"]) for kept in base_dets):
            discarded += 1
            continue
        det = copy.deepcopy(det)
        det["origin_record_id"] = incoming_id
        added.append(det)
    detections = base_dets + added
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
    return complete_record(record_id, snapshot, result, {"base": base["parameters"], "incoming": incoming["parameters"]},
                           kind="merge", merge={"base_id": base_id, "incoming_id": incoming_id,
                                                "discarded_overlaps": discarded, "added_count": len(added),
                                                "rule": "any_positive_overlap_keep_base"})
