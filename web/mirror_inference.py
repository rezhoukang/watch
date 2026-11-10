"""Original and horizontal mirror target-detection comparisons."""
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import aggregate
import aggregate_batch
import aggregate_jobs
from project_paths import IMAGE_EXTENSIONS
from . import common, image_folders

router = APIRouter(prefix='/api/mirror')


class MirrorRequest(BaseModel):
    model_config = {'extra': 'forbid'}
    filenames: list[str] = Field(min_length=1, max_length=1000)


@router.post('/batch')
def start_mirror(payload: MirrorRequest):
    sources = [common._safe_file(common.UPLOAD_DIR, name) for name in dict.fromkeys(payload.filenames)]
    if any(source.suffix.lower() not in IMAGE_EXTENSIONS for source in sources):
        raise HTTPException(status_code=400, detail='镜像推理只支持图片')
    common._require_head_detection()
    paths = image_folders.paths()
    try:
        return aggregate_batch.start(sources, False, mirror=True,
                                     relative_paths=[paths.get(source.name, source.name) for source in sources])
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get('/active')
def active_mirror():
    job = aggregate_jobs.active()
    return {'job': job if job and job.get('mirror') else None}


@router.get('/history')
def mirror_history():
    rows = []
    for path in (aggregate.RESULT_DIR / 'batches').glob('*/record.json'):
        try:
            job = aggregate_batch.saved(path.parent.name)
            if not job.get('mirror'):
                continue
            for index, row in enumerate(job['results']):
                if row['status'] == 'completed' and row.get('mirror_result'):
                    rows.append({'id': f'{job["job_id"]}-{index}', 'source': row['source'],
                                 'relative_path': row.get('relative_path', row['source']),
                                 'created_at': row['result'].get('generated_at', ''),
                                 'count': row['result']['count'], 'mirror_count': row['mirror_result']['count'],
                                 'thumbnail': row['result']['result_url']})
        except (OSError, ValueError, KeyError):
            continue
    return {'records': sorted(rows, key=lambda row: row['created_at'], reverse=True)}


@router.get('/history/{record_id}')
def mirror_record(record_id: str):
    if not re.fullmatch(r'[a-f0-9]{32}-[0-9]+', record_id):
        raise HTTPException(status_code=404, detail='镜像结果不存在')
    job_id, index = record_id.split('-')
    try:
        job = aggregate_batch.saved(job_id)
        row = job['results'][int(index)]
        if not job.get('mirror') or row['status'] != 'completed' or not row.get('mirror_result'):
            raise FileNotFoundError()
        return row
    except (FileNotFoundError, IndexError, ValueError):
        raise HTTPException(status_code=404, detail='镜像结果不存在') from None
