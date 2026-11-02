"""Background aggregate jobs with real stage/target progress and bounded in-memory history."""
import copy
import threading
import time
import uuid
from pathlib import Path

import aggregate

LABELS = {'prepare': '准备图片与模型', 'detect': '目标检测', 'face': '人脸识别',
          'pose': 'Pose 姿势检测', 'multimodal': 'VLM 动作分析', 'render': '生成结果'}
_lock = threading.RLock()
_runner = threading.Lock()
JOBS: dict[str, dict] = {}
TERMINAL = {'completed', 'failed'}


def get(job_id: str) -> dict:
    with _lock:
        if job_id not in JOBS:
            raise FileNotFoundError('聚合任务不存在或服务已重启，请重新推理')
        return copy.deepcopy(JOBS[job_id])


def active() -> dict | None:
    with _lock:
        return next((copy.deepcopy(j) for j in JOBS.values() if j['status'] not in TERMINAL), None)


def start(src: Path, options: dict) -> dict:
    if not _runner.acquire(blocking=False):
        raise RuntimeError('已有聚合任务正在运行，请等待完成')
    job_id = uuid.uuid4().hex
    plan = ['prepare', 'detect']
    if options.get('face', True):
        plan.append('face')
    if options.get('pose', True):
        plan.append('multimodal' if options.get('multimodal') else 'pose')
    plan.append('render')
    with _lock:
        # Keep at most 20 completed jobs, including their structured results.
        for old in list(JOBS):
            if len(JOBS) < 20:
                break
            if JOBS[old]['status'] in TERMINAL:
                del JOBS[old]
        JOBS[job_id] = {'job_id': job_id, 'source': src.name, 'status': 'queued',
                        'created_at': time.time(), 'elapsed_seconds': 0, 'stage': 'prepare',
                        'message': '任务已排队', 'current': 0, 'total': None, 'percent': 0,
                        'stages': [{'key': key, 'label': LABELS[key], 'status': 'waiting'} for key in plan]}

    def progress(event):
        with _lock:
            job = JOBS[job_id]
            key = event['stage']
            if key not in plan:
                return
            position = plan.index(key)
            total, current = event['total'], event['current']
            fraction = min(1.0, current / total) if total else (1.0 if total == 0 else 0.0)
            job.update(event)
            job['percent'] = round((position + fraction) / len(plan) * 100, 1)
            job['elapsed_seconds'] = round(time.time() - job['created_at'], 1)
            for index, stage in enumerate(job['stages']):
                stage['status'] = 'completed' if index < position or (index == position and fraction == 1) else (
                    'running' if index == position else 'waiting')

    def work():
        with _lock:
            JOBS[job_id]['status'] = 'running'
        try:
            result = aggregate.aggregate_image(src, **options, progress=progress)
            with _lock:
                JOBS[job_id].update(status='completed', result=result, percent=100, message='聚合完成')
                for stage in JOBS[job_id]['stages']:
                    stage['status'] = 'completed'
        except Exception as error:
            with _lock:
                job = JOBS[job_id]
                job.update(status='failed', error=str(error), message=str(error))
                for stage in job['stages']:
                    if stage['key'] == job['stage']:
                        stage['status'] = 'failed'
        finally:
            with _lock:
                JOBS[job_id]['elapsed_seconds'] = round(time.time() - JOBS[job_id]['created_at'], 1)
            _runner.release()

    try:
        threading.Thread(target=work, name=f'aggregate-{job_id[:8]}', daemon=True).start()
    except Exception:
        with _lock:
            del JOBS[job_id]
        _runner.release()
        raise
    return get(job_id)
