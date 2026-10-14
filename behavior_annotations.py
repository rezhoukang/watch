"""Behavior datasets reuse the YOLO annotator while retaining Pose sidecars."""
import copy
import json
import math
import re
import shutil
import threading
import uuid
from pathlib import Path

import dataset_store
import inference_history
import pose_behaviors
import pose_support
from project_paths import DATASETS_DIR

CLASSES = ["待标注", "低头", "抬头", "正常", "无法判断", "举手"]
DEFAULT_DATASET = "east_classroom_behavior"
_lock = threading.RLock()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def dataset_dir(name: str) -> Path:
    if not isinstance(name, str) or not re.fullmatch(r"[\w-]{1,80}", name) or Path(name).is_reserved():
        raise ValueError("数据集名称只能包含文字、数字、下划线或短横线")
    directory = DATASETS_DIR / name
    if directory.is_symlink() or directory.resolve().parent != DATASETS_DIR.resolve():
        raise ValueError("非法数据集路径")
    return directory


def is_behavior(directory: Path) -> bool:
    return (directory / "behavior_dataset.json").is_file()


def sidecar(image: Path) -> Path:
    return dataset_store.find_dataset_dir(image) / "behavior" / image.parent.name / (image.stem + ".json")


def read(image: Path) -> dict | None:
    path = sidecar(image)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def import_records(name: str, record_ids: list[str] | None = None) -> dict:
    directory = dataset_dir(name)
    if record_ids is None:
        latest = {}
        for item in inference_history.list_records():
            if item.get("mode") == "pose" and "东教室" in item["source"]:
                latest.setdefault(item["source"], item["record_id"])
        record_ids = list(latest.values())
    if not isinstance(record_ids, list) or not record_ids or any(not isinstance(v, str) for v in record_ids):
        raise ValueError("请选择 Pose 推理记录，或先完成东教室图片 Pose 推理")
    records = [inference_history.read_record(v) for v in dict.fromkeys(record_ids)]
    for record in records:
        if record.get("mode") != "pose" or any(len(d.get("keypoints", [])) != 17 for d in record["detections"]):
            raise ValueError("只能导入含 17 个关键点的 Pose 推理记录")
        if not inference_history.source_path(record).is_file():
            raise ValueError("推理原图不存在")
    with _lock:
        if directory.exists() and not is_behavior(directory):
            raise ValueError("同名普通数据集已存在，请为行为标注使用新的数据集名称")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "images" / "val").mkdir(parents=True, exist_ok=True)
        dataset_store.write_dataset_classes(directory, CLASSES)
        _write(directory / "behavior_dataset.json", {"classes": CLASSES, "hand_raise_rule": pose_behaviors.RULE})
        imported, skipped = 0, 0
        for record in records:
            filename = f'{Path(record["source"]).stem[:60]}__{record["record_id"]}{Path(record["source"]).suffix}'
            image = directory / "images" / "train" / filename
            if sidecar(image).is_file():
                skipped += 1
                continue
            image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inference_history.source_path(record), image)
            detections = []
            for index, original in enumerate(record["detections"]):
                if original["confidence"] <= 0.7:
                    continue
                det = copy.deepcopy(original)
                behavior = pose_behaviors.classify(det["keypoints"], record["image_width"], record["image_height"])
                # Keep head-state labels pending for independent human calibration.
                state = "举手" if behavior["state"] == "举手" else "待标注"
                det.update({"sample_id": f'{record["record_id"]}:{index}', "pose_bbox": original["bbox"],
                            "class_id": CLASSES.index(state), "class_name": state, "label": state,
                            "label_source": "rule" if state == "举手" else "pending", "behavior": behavior})
                detections.append(det)
            payload = {"source_record_id": record["record_id"], "source": record["source"],
                       "parameters": record["parameters"], "width": record["image_width"], "height": record["image_height"],
                       "pose": pose_support.metadata(), "detections": detections}
            dataset_store.save_labels_for_image(image, detections)
            _write(sidecar(image), payload)
            imported += 1
        return {"ok": True, "dataset": name, "imported": imported, "skipped": skipped, "progress": progress(directory)}


def save(image: Path, detections: list[dict]) -> tuple[int, str]:
    with _lock:
        original = read(image)
        if original is None:
            raise ValueError("行为标注元数据不存在，请重新导入 Pose 结果")
        known = {d["sample_id"]: d for d in original["detections"]}
        saved, seen = [], set()
        for incoming in detections:
            cid, bbox = incoming.get("class_id"), incoming.get("bbox")
            if isinstance(cid, bool) or not isinstance(cid, int) or not 0 <= cid < len(CLASSES):
                raise ValueError("每人必须选择一个有效的行为状态")
            if not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox):
                raise ValueError("人体框坐标无效")
            if not (0 <= bbox[0] < bbox[2] <= original["width"] and 0 <= bbox[1] < bbox[3] <= original["height"]):
                raise ValueError("人体框必须在图像范围内")
            sid = incoming.get("sample_id")
            if sid is not None and (not isinstance(sid, str) or sid not in known or sid in seen):
                raise ValueError("人员编号无效或重复，请重新打开标注器")
            sid = sid or uuid.uuid4().hex
            seen.add(sid)
            det = copy.deepcopy(known.get(sid, {"sample_id": sid, "confidence": None, "keypoints": []}))
            unchanged = det.get("class_id") == cid
            det.update({"bbox": bbox, "class_id": cid, "class_name": CLASSES[cid], "label": CLASSES[cid],
                        "label_source": det.get("label_source", "human") if unchanged else "human"})
            saved.append(det)
        count, relative = dataset_store.save_labels_for_image(image, saved)
        _write(sidecar(image), {**original, "detections": saved})
        return count, relative


def progress(directory: Path) -> dict:
    counts = {name: 0 for name in CLASSES}
    images = 0
    for path in (directory / "behavior").glob("*/*.json"):
        images += 1
        for det in json.loads(path.read_text(encoding="utf-8"))["detections"]:
            counts[det["class_name"]] += 1
    total = sum(counts.values())
    return {"images": images, "total": total, "labeled": total-counts["待标注"], "counts": counts}


def export(name: str) -> dict:
    directory = dataset_dir(name)
    if not is_behavior(directory):
        raise ValueError("请选择行为标注数据集")
    with _lock:
        images = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((directory / "behavior").glob("*/*.json"))]
        return {"dataset": name, "classes": CLASSES, "hand_raise_rule": pose_behaviors.RULE,
                "progress": progress(directory), "images": images}
