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
    """聚合参数：只有两个开关。

    目标检测、人脸识别阈值、姿势配置、绑定规则全部锁死在 aggregate.py 里（
    FACE_CONFIDENCE / MATCH_THRESHOLD / MIN_QUALITY_CONFIDENCE / MIN_FACE_PIXELS /
    MIN_FACE_SHARPNESS、POSE_* 、MIN_BIND_IOU），前端不开放也不提交。
    """
    face: bool = True
    pose: bool = True


@router.post('/api/aggregate/image/{filename}', summary='聚合操作：目标检测 + 人脸识别 + 姿势检测')
def aggregate_image(filename: str, payload: AggregateRequest) -> dict:
    common._require_head_detection()
    src = common._safe_file(common.UPLOAD_DIR, filename)
    if src.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail='只能聚合待推理列表中的图片')
    common.logger.info('开始聚合：%s（人脸=%s, 姿势=%s）', filename, payload.face, payload.pose)
    try:
        result = aggregate.aggregate_image(src, face=payload.face, pose=payload.pose)
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
