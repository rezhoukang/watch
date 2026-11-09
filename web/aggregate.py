"""聚合操作：单图 目标检测 + 人脸识别 + 姿势检测绑定的接口。"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import aggregate
import aggregate_jobs
import aggregate_batch
import aggregate_history
import face_recognition
from project_paths import IMAGE_EXTENSIONS
from . import common

router = APIRouter()


class AggregateRequest(BaseModel):
    """仅保留人脸与多模态姿态开关，多模态固定采用两阶段策略一。"""
    face: bool = True
    multimodal: bool = False

    def options(self):
        return {**self.model_dump(), 'pose': False}


class BatchRequest(BaseModel):
    model_config = {'extra': 'forbid'}
    filenames: list[str] = Field(min_length=1, max_length=1000)
    multimodal: bool = False


class HistorySelection(BaseModel):
    record_ids: list[str] = Field(min_length=1, max_length=1000)


@router.get('/api/aggregate/history')
def history_list():
    return {'records': aggregate_history.listing()}


@router.get('/api/aggregate/history/{record_id}')
def history_record(record_id: str):
    try:
        return aggregate_history.get(record_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post('/api/aggregate/history/delete')
def history_delete(payload: HistorySelection):
    return aggregate_history.delete(payload.record_ids)


@router.post('/api/aggregate/history/export')
def history_export(payload: HistorySelection):
    try:
        return Response(aggregate_history.export(payload.record_ids), media_type='application/zip', headers={
            'Content-Disposition': 'attachment; filename="aggregate-results.zip"'})
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post('/api/aggregate/batch', summary='批量目标检测与多模态姿态识别')
def start_batch(payload: BatchRequest):
    common._require_head_detection()
    sources = [common._safe_file(common.UPLOAD_DIR, name) for name in dict.fromkeys(payload.filenames)]
    if any(path.suffix.lower() not in IMAGE_EXTENSIONS for path in sources):
        raise HTTPException(status_code=400, detail='只能批量聚合图片')
    try:
        return aggregate_batch.start(sources, payload.multimodal)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get('/api/aggregate/batches/{job_id}/{kind}/{filename}')
def batch_image(job_id: str, kind: str, filename: str):
    if kind not in ('sources', 'results'):
        raise HTTPException(status_code=404, detail='图片不存在')
    try:
        root = aggregate_batch.directory(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(common._safe_file(root / kind, filename))


@router.post('/api/aggregate/jobs/{filename}', summary='启动带阶段进度的聚合推理')
def start_aggregate_job(filename: str, payload: AggregateRequest):
    common._require_head_detection()
    src = common._safe_file(common.UPLOAD_DIR, filename)
    if src.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail='只能聚合待推理列表中的图片')
    try:
        return aggregate_jobs.start(src, payload.options())
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get('/api/aggregate/jobs/active', summary='查询正在运行的聚合任务')
def active_aggregate_job():
    return {'job': aggregate_jobs.active()}


@router.get('/api/aggregate/jobs/{job_id}', summary='查询聚合阶段与进度')
def aggregate_job_status(job_id: str):
    try:
        return aggregate_jobs.get(job_id)
    except FileNotFoundError as error:
        try:
            return aggregate_batch.saved(job_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=str(error)) from error


@router.post('/api/aggregate/jobs/{job_id}/cancel', summary='停止聚合推理')
def cancel_aggregate_job(job_id: str):
    try:
        return aggregate_jobs.cancel(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post('/api/aggregate/image/{filename}', summary='聚合操作：目标检测 + 人脸识别 + 姿势检测')
def aggregate_image(filename: str, payload: AggregateRequest) -> dict:
    common._require_head_detection()
    src = common._safe_file(common.UPLOAD_DIR, filename)
    if src.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail='只能聚合待推理列表中的图片')
    common.logger.info('开始聚合：%s（人脸=%s, 多模态=%s）', filename, payload.face, payload.multimodal)
    try:
        result = aggregate.aggregate_image(src, **payload.options())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
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
