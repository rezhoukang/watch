"""Atomic, persistent records and image snapshots for independent action runs."""
import copy
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from project_paths import OUTPUT_DIR
from .regions import Expansion, estimate, render_crop, validate_box

ROOT = OUTPUT_DIR / "action_recognition"
_lock = threading.RLock()
_RUN_ID = re.compile(r"^[a-f0-9]{32}$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def folder(run_id: str) -> Path:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("无效的动作识别批次编号")
    return ROOT / run_id


def read(run_id: str) -> dict:
    with _lock:
        return json.loads((folder(run_id) / "record.json").read_text(encoding="utf-8"))


def write(record: dict) -> None:
    with _lock:
        directory = folder(record["run_id"])
        directory.mkdir(parents=True, exist_ok=True)
        record["updated_at"] = now()
        temp = directory / "record.json.tmp"
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(directory / "record.json")


def create(names: list[str], expansion: Expansion) -> dict:
    record = {"run_id": uuid.uuid4().hex, "created_at": now(), "status": "preparing",
              "filenames": names, "expansion": expansion.to_dict(), "images": [],
              "crops": [], "errors": [], "crop_count": 0}
    write(record)
    return record


def list_runs() -> list[dict]:
    if not ROOT.is_dir():
        return []
    records = []
    for path in ROOT.glob("*/record.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            records.append({key: value[key] for key in
                            ("run_id", "created_at", "status", "filenames", "crop_count")})
        except (OSError, ValueError, KeyError):
            continue
    return sorted(records, key=lambda value: value["created_at"], reverse=True)


def snapshot(record: dict, source: Path, index: int) -> tuple[Path, Image.Image]:
    directory = folder(record["run_id"]) / "sources"
    directory.mkdir(exist_ok=True)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    path = directory / f"{index:03d}.png"
    image.save(path)
    return path, image


def save_crop(record: dict, image: Image.Image, item: dict, box: list) -> None:
    """Editing a region invalidates its previous action and cached inference."""
    region = validate_box(box, *image.size)
    hx1, hy1, hx2, hy2 = item["head_bbox"]
    if not (region[0] <= hx1 and region[1] <= hy1 and region[2] >= hx2 and region[3] >= hy2):
        raise ValueError("区域必须包含完整的目标头框")
    directory = folder(record["run_id"])
    (directory / "crops").mkdir(exist_ok=True)
    raw, marked = render_crop(image, region, item["head_bbox"])
    item["revision"] = item.get("revision", 0)+1
    left, top, right, bottom = region
    item.update({"region_bbox": region, "roi_polygon": [[left, top], [right, top], [right, bottom], [left, bottom]],
                 "roi_shape": "head_context_rectangle_v1", "raw_file": f"crops/{item['crop_id']}.png",
                 "marked_file": f"crops/{item['crop_id']}_target.png", "status": "pending",
                 "action": None, "analysis_key": None})
    item.pop("error", None)
    item.pop("elapsed_seconds", None)
    raw.save(directory / item["raw_file"])
    marked.save(directory / item["marked_file"])


def append_image(record: dict, source_name: str, index: int, image: Image.Image,
                 detections: list[dict], expansion: Expansion) -> None:
    info = {"source": source_name, "image_index": index, "width": image.width,
            "height": image.height, "source_file": f"sources/{index:03d}.png"}
    record["images"].append(info)
    for number, detection in enumerate(detections, 1):
        crop_id = f"HC{index:03d}-{number:03d}"
        try:
            head = validate_box(detection["bbox"], *image.size)
            item = {"crop_id": crop_id, "image_index": index, "source": source_name,
                    "source_file": info["source_file"], "head_bbox": head,
                    "head_confidence": detection["confidence"]}
            save_crop(record, image, item, estimate(head, *image.size, expansion))
            record["crops"].append(item)
        except (KeyError, ValueError) as error:
            record["errors"].append({"source": source_name, "crop_id": crop_id, "message": str(error)})
    record["crop_count"] = len(record["crops"])


def find_crop(record: dict, crop_id: str) -> dict:
    for item in record["crops"]:
        if item["crop_id"] == crop_id:
            return item
    raise FileNotFoundError("目标裁剪不存在")


def update_region(run_id: str, crop_id: str, box: list) -> dict:
    record = read(run_id)
    item = find_crop(record, crop_id)
    with Image.open(folder(run_id) / item["source_file"]) as image:
        save_crop(record, image, item, box)
    write(record)
    return copy.deepcopy(record)


def reexpand(run_id: str, expansion: Expansion) -> dict:
    record = read(run_id)
    # Stage the JSON update; all coordinates are validated before writing crops.
    for info in record["images"]:
        with Image.open(folder(run_id) / info["source_file"]) as image:
            for item in record["crops"]:
                if item["image_index"] == info["image_index"]:
                    save_crop(record, image, item, estimate(item["head_bbox"], *image.size, expansion))
    record["expansion"] = expansion.to_dict()
    write(record)
    return record


def image_file(run_id: str, filename: str) -> Path:
    directory = folder(run_id).resolve()
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory) or path.suffix.lower() != ".png" or not path.is_file():
        raise FileNotFoundError("动作识别图片不存在")
    return path
