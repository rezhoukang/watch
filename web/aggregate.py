"""聚合操作：单图 目标检测 + 人脸识别 + 姿势检测绑定的接口。"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import aggregate
import face_recognition
import vision
from project_paths import IMAGE_EXTENSIONS
from . import common

router = APIRouter()


class AggregateRequest(BaseModel):
    """聚合参数。人脸识别 / 姿势检测用勾选框控制是否参与。"""
    weights: str = vision.DEFAULT_WEIGHTS
    confidence: float | None = Field(default=None, ge=0.01, le=0.99)
    imgsz: int = Field(default=640, ge=320, le=1920)
    classes: str | None = None
    face: bool = True
    pose: bool = True
    # 人脸识别
    threshold: float = Field(default=face_recognition.DEFAULT_MATCH_THRESHOLD, ge=0, le=1)
    face_confidence: float = Field(default=face_recognition.DETECT_CONFIDENCE, ge=0.01, le=0.99)
    min_confidence: float = Field(default=face_recognition.DEFAULT_QUALITY_CONFIDENCE, ge=0, le=1)
    min_size: int = Field(default=face_recognition.DEFAULT_MIN_FACE_PIXELS, ge=1, le=1000)
    min_sharpness: float = Field(default=face_recognition.DEFAULT_MIN_SHARPNESS, ge=0, le=10000)
    # 姿势检测与绑定
    pose_confidence: float = Field(default=aggregate.DEFAULT_POSE_CONFIDENCE, ge=0.01, le=0.99)
    bind: str = aggregate.DEFAULT_BIND_STRATEGY
    bind_iou: float = Field(default=aggregate.DEFAULT_BIND_IOU, ge=0, le=1)


@router.post('/api/aggregate/image/{filename}', summary='聚合操作：目标检测 + 人脸识别 + 姿势检测')
def aggregate_image(filename: str, payload: AggregateRequest) -> dict:
    common._require_image_backend(payload.weights)
    src = common._safe_file(common.UPLOAD_DIR, filename)
    if src.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail='只能聚合待推理列表中的图片')
    conf = common._clamp_conf(payload.confidence if payload.confidence is not None
                              else vision.resolve_confidence(payload.weights))
    common.logger.info('开始聚合：%s（权重=%s, 人脸=%s, 姿势=%s, 绑定=%s）',
                       filename, payload.weights, payload.face, payload.pose, payload.bind)
    try:
        result = aggregate.aggregate_image(
            src, weights=payload.weights, conf=conf, classes=common._parse_classes(payload.classes),
            imgsz=payload.imgsz, face=payload.face, pose=payload.pose,
            threshold=payload.threshold, face_confidence=payload.face_confidence,
            min_confidence=payload.min_confidence, min_size=payload.min_size,
            min_sharpness=payload.min_sharpness, pose_confidence=payload.pose_confidence,
            bind=payload.bind, bind_iou=payload.bind_iou)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        common.logger.exception('聚合失败：%s', filename)
        raise HTTPException(status_code=500, detail=f'聚合失败（{type(exc).__name__}）：{exc}') from exc
    common.logger.info('聚合完成：%s 目标 %s 个，命中姓名 %s 个，耗时 %ss',
                       filename, result['count'], result['summary']['named_targets'],
                       result['elapsed_seconds'])
    return result


@router.get('/api/aggregate/results/{filename}', summary='访问聚合结果图')
def aggregate_result(filename: str):
    return FileResponse(common._safe_file(aggregate.RESULT_DIR, filename))
