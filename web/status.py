"""status: extracted API routes; URLs and response formats are unchanged."""
from fastapi import APIRouter
import vision
import rtdetrv4
from . import common

router = APIRouter()


# ---------- YOLO 状态 ----------
@router.get("/api/yolo/status", summary="YOLO 环境状态")
def yolo_status() -> dict:
    available, message = vision.yolo_available()
    weights: list[str] = []
    world_weights: list[str] = []
    if common.MODELS_DIR.is_dir():
        for p in sorted(common.MODELS_DIR.glob("*.pt")):
            weights.append(p.name)
            if vision.is_world_weights(p.name):
                world_weights.append(p.name)
    head_model = rtdetrv4.status()
    return {
        "available": available,
        "message": message,
        "weights": weights,
        "world_weights": world_weights,             # 属于 YOLO-World 的权重
        "default_weights": vision.DEFAULT_WEIGHTS,  # 前端下拉框默认选中它
        "clip_available": vision.clip_available(),  # World 所需的文本编码器依赖
        "image_weights": weights + [rtdetrv4.WEIGHTS_NAME],
        "default_image_weights": rtdetrv4.WEIGHTS_NAME if head_model["available"] else vision.DEFAULT_WEIGHTS,
        "model_labels": {rtdetrv4.WEIGHTS_NAME: rtdetrv4.MODEL_NAME},
        "rtdetrv4": head_model,
    }


@router.get("/api/rtdetrv4/status", summary="RT-DETRv4 人头检测环境状态")
def rtdetrv4_status() -> dict:
    return rtdetrv4.status()
