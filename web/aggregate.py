"""聚合操作：单图 目标检测 + 人脸识别 + 姿势检测绑定的接口。"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import aggregate
import face_recognition
from project_paths import IMAGE_EXTENSIONS
from . import common

router = APIRouter()


class AggregateRequest(BaseModel):
    """聚合参数：目标检测与绑定规则都不在这里配。

    目标检测固定复用「图片推理 → 目标检测」；绑定固定为链式贪心 + 一一对应（IoU > 0）。
    人脸识别 / 姿势检测用勾选框控制是否参与。
    """
    face: bool = True
    pose: bool = True
    # 人脸识别
    threshold: float = Field(default=face_recognition.DEFAULT_MATCH_THRESHOLD, ge=0, le=1)
    face_confidence: float = Field(default=face_recognition.DETECT_CONFIDENCE, ge=0.01, le=0.99)
    min_confidence: float = Field(default=face_recognition.DEFAULT_QUALITY_CONFIDENCE, ge=0, le=1)
    min_size: int = Field(default=face_recognition.DEFAULT_MIN_FACE_PIXELS, ge=1, le=1000)
    min_sharpness: float = Field(default=face_recognition.DEFAULT_MIN_SHARPNESS, ge=0, le=10000)
    # 姿势检测（绑定规则固定，不可配）
    pose_confidence: float = Field(default=aggregate.DEFAULT_POSE_CONFIDENCE, ge=0.01, le=0.99)


@router.post('/api/aggregate/image/{filename}', summary='聚合操作：目标检测 + 人脸识别 + 姿势检测')
def aggregate_image(filename: str, payload: AggregateRequest) -> dict:
    common._require_head_detection()
    src = common._safe_file(common.UPLOAD_DIR, filename)
    if src.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail='只能聚合待推理列表中的图片')
    common.logger.info('开始聚合：%s（人脸=%s, 姿势=%s）', filename, payload.face, payload.pose)
    try:
        result = aggregate.aggregate_image(
            src, face=payload.face, pose=payload.pose,
            threshold=payload.threshold, face_confidence=payload.face_confidence,
            min_confidence=payload.min_confidence, min_size=payload.min_size,
            min_sharpness=payload.min_sharpness, pose_confidence=payload.pose_confidence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        common.logger.exception('聚合失败：%s', filename)
        raise HTTPException(status_code=500, detail=f'聚合失败（{type(exc).__name__}）：{exc}') from exc
    common.logger.info('聚合完成：%s 人头 %s 个，命中姓名 %s 个，耗时 %ss',
                       filename, result['count'], result['summary']['named_targets'],
                       result['elapsed_seconds'])
    return result


@router.get('/api/aggregate/results/{filename}', summary='访问聚合结果图')
def aggregate_result(filename: str):
    return FileResponse(common._safe_file(aggregate.RESULT_DIR, filename))
