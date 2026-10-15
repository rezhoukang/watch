"""Batch target-detection crops and persistent head-management records."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import head_cuts
from project_paths import IMAGE_EXTENSIONS
from . import common

router = APIRouter()


class BatchRequest(BaseModel):
    filenames: list[str] = Field(min_length=1, max_length=200)
    rtdetr_base: bool = False
    merge_iou: float = Field(default=.1, ge=0, le=1)
    merge_confidence: float = Field(default=.55, ge=0, le=1)


@router.post('/api/head-cuts')
def start_batch(payload: BatchRequest) -> dict:
    paths = [common._safe_file(common.UPLOAD_DIR, name) for name in dict.fromkeys(payload.filenames)]
    if any(p.suffix.lower() not in IMAGE_EXTENSIONS or p.resolve().parent != common.UPLOAD_DIR.resolve() for p in paths):
        raise HTTPException(status_code=400, detail='只能选择待推理列表中的图片')
    common._require_head_detection()
    run_id = head_cuts.start_batch(paths, payload.rtdetr_base, payload.merge_iou, payload.merge_confidence)
    return {'run_id': run_id, 'job_id': run_id}


@router.get('/api/head-cuts')
def list_batches() -> dict:
    return {'runs': head_cuts.list_runs()}


@router.get('/api/head-cuts/{run_id}')
def get_batch(run_id: str) -> dict:
    try:
        return head_cuts.read_run(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail='头切割批次不存在') from exc


@router.post('/api/head-cuts/{run_id}/marks')
def save_marks(run_id: str, payload: dict) -> dict:
    get_batch(run_id)
    try:
        head_cuts.save_marks(run_id, payload.get('marks', {}))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {'ok': True}


@router.get('/api/head-cuts/{run_id}/download')
def download(run_id: str):
    get_batch(run_id)
    try:
        path = head_cuts.build_archive(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, filename=f'head_cuts_{run_id[:8]}.zip')


@router.get('/api/head-cuts/{run_id}/files/{filename:path}')
def get_file(run_id: str, filename: str):
    get_batch(run_id)
    folder = head_cuts.run_dir(run_id).resolve()
    path = (folder/filename).resolve()
    if not path.is_relative_to(folder) or not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail='裁剪图片不存在')
    return FileResponse(path)
