"""聚合操作：单图 目标检测 + 人脸识别 + 姿势检测绑定的接口。"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from urllib.parse import quote
from typing import Literal

import aggregate
import aggregate_jobs
import aggregate_batch
import aggregate_history
import face_recognition
from project_paths import IMAGE_EXTENSIONS
from . import common
from . import image_folders

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
    record_ids: list[str] = Field(default_factory=list, max_length=1000)
    folder_ids: list[str] = Field(default_factory=list, max_length=1000)


class HistoryExport(BaseModel):
    record_ids: list[str] = Field(default_factory=list, max_length=1000)
    folder_ids: list[str] = Field(default_factory=list, max_length=1000)
    compression: Literal['original', 'balanced', 'compact'] = 'original'


class HistoryRerun(HistorySelection):
    multimodal: bool = False


class FolderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    parent_id: str | None = None


class FolderMove(BaseModel):
    record_ids: list[str] = Field(default_factory=list, max_length=1000)
    folder_ids: list[str] = Field(default_factory=list, max_length=1000)
    folder_id: str | None = None


@router.get('/api/aggregate/folders')
def history_folders():
    aggregate_history.sync_source_folders(image_folders.paths())
    return {'folders': aggregate_history.folders()}


@router.post('/api/aggregate/folders')
def history_create_folder(payload: FolderCreate):
    try:
        return aggregate_history.create_folder(payload.name, payload.parent_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post('/api/aggregate/folders/move')
def history_move_folder(payload: FolderMove):
    try:
        if not payload.record_ids and not payload.folder_ids:
            raise HTTPException(status_code=422, detail='请选择记录或文件夹')
        aggregate_history.move_to_folder(payload.record_ids, payload.folder_id, payload.folder_ids)
        return {'ok': True}
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get('/api/aggregate/history')
def history_list():
    return {'records': aggregate_history.sync_source_folders(image_folders.paths())}


@router.get('/api/aggregate/history/{record_id}')
def history_record(record_id: str):
    try:
        return aggregate_history.get(record_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post('/api/aggregate/history/delete')
def history_delete(payload: HistorySelection):
    if not payload.record_ids and not payload.folder_ids:
        raise HTTPException(status_code=422, detail='请选择记录或文件夹')
    return aggregate_history.delete(payload.record_ids, payload.folder_ids)


@router.post('/api/aggregate/history/export')
def history_export(payload: HistoryExport):
    try:
        if not payload.record_ids and not payload.folder_ids:
            raise HTTPException(status_code=422, detail='请选择记录或文件夹')
        content, filename = aggregate_history.export_selection(payload.record_ids, payload.folder_ids, payload.compression)
        return Response(content, media_type='application/zip', headers={
            'Content-Disposition': f"attachment; filename=\"aggregate-results.zip\"; filename*=UTF-8''{quote(filename, safe='')}"})
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post('/api/aggregate/history/rerun', summary='使用保存的原图重新聚合推理')
def history_rerun(payload: HistoryRerun):
    try:
        records = aggregate_history.selected_records(payload.record_ids, payload.folder_ids)
        if not records:
            raise HTTPException(status_code=422, detail='请选择包含结果记录的图片或文件夹')
        sources = [common._safe_file(aggregate_batch.directory(row['id'].split('-')[0]) / 'sources',
                                     row['source_file']) for row in records]
        common._require_head_detection()
        paths = image_folders.paths()
        return aggregate_batch.start(sources, payload.multimodal, display_names=[row['source'] for row in records],
                                     relative_paths=[row.get('relative_path') or paths.get(
                                         row['source_file'], paths.get(row['source'], row['source']))
                                                     for row in records])
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post('/api/aggregate/batch', summary='批量目标检测与多模态姿态识别')
def start_batch(payload: BatchRequest):
    common._require_head_detection()
    sources = [common._safe_file(common.UPLOAD_DIR, name) for name in dict.fromkeys(payload.filenames)]
    if any(path.suffix.lower() not in IMAGE_EXTENSIONS for path in sources):
        raise HTTPException(status_code=400, detail='只能批量聚合图片')
    try:
        paths = image_folders.paths()
        return aggregate_batch.start(sources, payload.multimodal,
                                     relative_paths=[paths.get(source.name, source.name) for source in sources])
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get('/api/aggregate/batches/{job_id}/{kind}/{filename}')
def batch_image(job_id: str, kind: str, filename: str):
    if kind not in ('sources', 'results', 'mirrors', 'mirror-results'):
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
