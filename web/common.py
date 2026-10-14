"""Shared API paths, validation and upload helpers."""
import logging
import re
import shutil
from pathlib import Path

from fastapi import HTTPException, UploadFile

import vision
import rtdetrv4

logger = logging.getLogger("watch")


def _validate_head_thresholds(down: float, up: float) -> None:
    import pose_behaviors

    try:
        pose_behaviors.validate_thresholds(down, up)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

from project_paths import (
    STATIC_DIR, UPLOAD_DIR, VIDEO_DIR, RESULTS_DIR, OUTPUT_DIR, MODELS_DIR,
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS,
)

# Windows 文件名非法字符；同时挡掉路径分隔符，防目录穿越
_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Windows 保留设备名，不能直接当文件名
_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _safe_name(original: str, ext_set: set[str]) -> str:
    """沿用上传时的原文件名（只做安全清洗），扩展名必须在白名单内"""
    name = Path(original or "").name.strip()      # 去掉任何路径部分
    suffix = Path(name).suffix
    if suffix.lower() not in ext_set:
        raise HTTPException(
            status_code=400,
            detail=(
                f"不支持的文件格式 {suffix or '(无扩展名)'}，"
                f"仅支持：{', '.join(sorted(ext_set))}"
            ),
        )
    stem = _ILLEGAL_CHARS.sub("_", Path(name).stem).strip(" .")  # 结尾不能是点/空格
    if not stem:
        raise HTTPException(status_code=400, detail=f"文件名不合法：{original}")
    if stem.upper() in _RESERVED_NAMES:
        stem = f"_{stem}"
    return f"{stem}{suffix}"


def _save_upload(file: UploadFile, directory: Path, ext_set: set[str]) -> tuple[str, bool]:
    """按原文件名保存（同名直接覆盖）。返回 (文件名, 是否覆盖了已有文件)"""
    filename = _safe_name(file.filename or "", ext_set)
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / filename
    overwritten = dest.exists()
    try:
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"保存文件失败: {e}")
    return filename, overwritten


def _safe_file(directory: Path, filename: str) -> Path:
    """防目录穿越地取文件"""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    path = directory / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return path


def _parse_classes(classes: str | None) -> list[str] | None:
    """把 "person, car, 书本" 解析成类别列表（中英文逗号都支持），空则返回 None"""
    if not classes:
        return None
    items = [c.strip() for c in classes.replace("，", ",").split(",")]
    items = [c for c in items if c]
    return items or None


def _clamp_conf(conf: float) -> float:
    """置信度钳制到 (0, 1)：ultralytics 对越界值会直接抛 ValueError（如 conf=111）"""
    return max(0.01, min(conf, 0.99))


def _require_image_backend(weights: str) -> None:
    if rtdetrv4.is_weights(weights):
        state = rtdetrv4.status()
        if not state["available"]:
            raise HTTPException(status_code=503, detail=state["message"])
        return
    available, message = vision.yolo_available()
    if not available:
        raise HTTPException(status_code=503, detail=f"YOLO 不可用：{message}")
    ok, reason = vision.world_ready(weights)
    if not ok:
        raise HTTPException(status_code=503, detail=reason)


def _require_yolo_weights(weights: str) -> None:
    if rtdetrv4.is_weights(weights):
        raise HTTPException(status_code=400, detail="此操作暂不支持 RT-DETRv4，请在图片或视频推理页使用")


def _require_head_detection() -> None:
    from head_fusion import BASE_WEIGHTS
    _require_image_backend(BASE_WEIGHTS)
    _require_image_backend(rtdetrv4.WEIGHTS_NAME)


def _delete_batch(items: list[str], delete_item) -> dict:
    deleted, missing, failed = [], [], []
    for item in dict.fromkeys(items):
        try:
            delete_item(item)
            deleted.append(item)
        except FileNotFoundError:
            missing.append(item)
        except (OSError, ValueError) as e:
            failed.append({"item": item, "detail": str(e)})
    return {"deleted": deleted, "missing": missing, "failed": failed}


def _delete_items(payload: dict, key: str) -> list[str]:
    items = payload.get(key)
    if (not isinstance(items, list) or not 1 <= len(items) <= 1000 or
            any(not isinstance(item, str) or not item for item in items)):
        raise HTTPException(status_code=400, detail="请选择 1~1000 项进行删除")
    return items
