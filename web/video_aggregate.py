"""MP4 sampling and video target/action aggregate endpoints."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import video_aggregate
from action_recognition import jobs, store
from . import common
from .actions import _call

router = APIRouter(prefix='/api/video-aggregate', tags=['视频聚合推理'])


class StartRequest(BaseModel):
    filename: str
    interval_seconds: float = Field(default=1, ge=0.1, le=86400, allow_inf_nan=False)


@router.post('/jobs')
def start(payload: StartRequest):
    common._require_head_detection()
    source = common._safe_file(common.VIDEO_DIR, payload.filename)
    if source.suffix.lower() != '.mp4':
        raise HTTPException(status_code=400, detail='视频聚合支持 MP4 文件')
    return _call(video_aggregate.start, source, payload.interval_seconds)


@router.get('/jobs/active')
def active():
    job = jobs.active()
    return {'job': job if job and job['kind'] == video_aggregate.KIND else None}


def video_job(job_id: str):
    job = jobs.get(job_id)
    if job['kind'] != video_aggregate.KIND:
        raise FileNotFoundError('视频聚合任务不存在')
    return job


@router.get('/jobs/{job_id}')
def status(job_id: str):
    return _call(video_job, job_id)


@router.post('/jobs/{job_id}/cancel')
def cancel(job_id: str):
    _call(video_job, job_id)
    return _call(jobs.cancel, job_id)


@router.get('/runs')
def list_runs():
    return {'runs': video_aggregate.history()}


@router.get('/runs/{run_id}')
def read_run(run_id: str):
    return _call(video_aggregate.describe, run_id)


@router.get('/runs/{run_id}/frames/{index}')
def frame_detail(run_id: str, index: int):
    record = _call(video_aggregate.read, run_id)
    frame = next((f for f in record['video']['frames'] if f['image_index'] == index), None)
    if frame is None:
        raise HTTPException(status_code=404, detail='采样帧不存在')
    return {**frame, 'targets': video_aggregate.targets(record, index)}


@router.get('/runs/{run_id}/video')
def original_video(run_id: str):
    _call(video_aggregate.read, run_id)
    return FileResponse(common._safe_file(store.folder(run_id), 'source.mp4'), media_type='video/mp4')


@router.get('/runs/{run_id}/files/{filename:path}')
def image(run_id: str, filename: str):
    _call(video_aggregate.read, run_id)
    return FileResponse(_call(store.image_file, run_id, filename), headers={'Cache-Control': 'no-store'})


@router.get('/runs/{run_id}/export')
def export(run_id: str):
    _call(video_aggregate.read, run_id)
    return FileResponse(store.folder(run_id) / 'record.json', media_type='application/json',
                        filename=f'video-aggregate-{run_id}.json')
