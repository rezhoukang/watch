"""Serial image batches for target detection and VLM actions."""
import json
import re
import shutil
import threading
import time
import uuid
import cv2
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import aggregate
import aggregate_jobs


def directory(job_id: str) -> Path:
    if not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise FileNotFoundError('批量聚合任务不存在')
    return aggregate.RESULT_DIR / 'batches' / job_id


def saved(job_id: str) -> dict:
    path = directory(job_id) / 'record.json'
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise FileNotFoundError('批量聚合任务不存在') from None
    if record['status'] not in aggregate_jobs.TERMINAL:
        record.update(status='failed', message='服务已重启，批次已中断，已完成结果可查看')
    return record


def persist(job_id: str) -> None:
    record = aggregate_jobs.get(job_id)
    root = directory(job_id)
    root.mkdir(parents=True, exist_ok=True)
    temp = root / 'record.json.tmp'
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(root / 'record.json')


def start(sources: list[Path], multimodal: bool, *, display_names: list[str] | None = None,
          relative_paths: list[str] | None = None, mirror: bool = False) -> dict:
    if not sources:
        raise ValueError('请选择需要批量推理的图片')
    if mirror and multimodal:
        raise ValueError('镜像推理仅支持目标检测')
    relative_paths = relative_paths if relative_paths is not None else [path.name for path in sources]
    names = display_names if display_names is not None else [PurePosixPath(path).name for path in relative_paths]
    if len(names) != len(sources) or len(relative_paths) != len(sources):
        raise ValueError('图片名称与来源数量不一致')
    if not aggregate_jobs._runner.acquire(blocking=False):
        raise RuntimeError('已有聚合任务正在运行，请等待完成')
    job_id = uuid.uuid4().hex
    plan = ['prepare', 'detect'] + (['multimodal'] if multimodal else []) + ['render']
    if mirror:
        plan += ['mirror_prepare', 'mirror_detect', 'mirror_render']
    with aggregate_jobs._lock:
        for old in list(aggregate_jobs.JOBS):
            if len(aggregate_jobs.JOBS) < 20:
                break
            if aggregate_jobs.JOBS[old]['status'] in aggregate_jobs.TERMINAL:
                del aggregate_jobs.JOBS[old]
        aggregate_jobs.JOBS[job_id] = {
            'job_id': job_id, 'batch': True, 'mirror': mirror, 'source': names[0], 'status': 'queued',
            'created_at': time.time(), 'elapsed_seconds': 0, 'stage': 'prepare', 'percent': 0,
            'message': '批量聚合已排队', 'current': 0, 'total': None,
            'image_index': 1, 'image_total': len(sources), 'completed_images': 0, 'failed_images': 0,
            'stages': [{'key': key, 'label': aggregate_jobs.LABELS[key], 'status': 'waiting'} for key in plan],
            'results': [{'source': name, 'source_file': f'{index:04d}{path.suffix}' if display_names is not None else path.name,
                         'relative_path': relative_paths[index],
                         'status': 'waiting'} for index, (path, name) in enumerate(zip(sources, names))],
            'options': {'face': False, 'pose': False, 'multimodal': multimodal},
        }

    def update(**values):
        with aggregate_jobs._lock:
            aggregate_jobs.JOBS[job_id].update(values)

    def progress(event):
        aggregate_jobs.check_cancelled(job_id)
        stage = event['stage']
        if stage not in plan:
            return
        total, current = event['total'], event['current']
        fraction = min(1, current / total) if total else (1 if total == 0 else 0)
        with aggregate_jobs._lock:
            job = aggregate_jobs.JOBS[job_id]
            position = plan.index(stage)
            job.update(event)
            job['percent'] = round(((job['image_index'] - 1) + (position + fraction) / len(plan)) / len(sources) * 100, 1)
            for index, step in enumerate(job['stages']):
                step['status'] = 'completed' if index < position or (index == position and fraction == 1) else 'running' if index == position else 'waiting'

    def work():
        update(status='running')
        root = directory(job_id)
        try:
            (root / 'sources').mkdir(parents=True, exist_ok=True)
            (root / 'results').mkdir(exist_ok=True)
            if mirror:
                (root / 'mirrors').mkdir()
                (root / 'mirror-results').mkdir()
            for index, source in enumerate(sources):
                aggregate_jobs.check_cancelled(job_id)
                update(source=names[index], image_index=index + 1, preview_targets=[],
                       stage='prepare', message=f'准备第 {index + 1}/{len(sources)} 张图片', current=0, total=None)
                with aggregate_jobs._lock:
                    job = aggregate_jobs.JOBS[job_id]
                    row = job['results'][index]
                    row['status'] = 'running'
                    for step in job['stages']:
                        step['status'] = 'waiting'
                persist(job_id)
                try:
                    snapshot = root / 'sources' / row['source_file']
                    shutil.copyfile(source, snapshot)
                    result = aggregate.aggregate_image(snapshot, face=False, pose=False,
                                                       multimodal=multimodal, progress=progress, mirror_selection=not mirror)
                    rendered = aggregate.RESULT_DIR / result['result']
                    destination = root / 'results' / rendered.name
                    shutil.copyfile(rendered, destination)
                    prefix = f'/api/aggregate/batches/{job_id}'
                    result['source_url'] = f'{prefix}/sources/{quote(snapshot.name)}'
                    result['result_url'] = f'{prefix}/results/{quote(destination.name)}'
                    if mirror:
                        aggregate_jobs.check_cancelled(job_id)
                        mirrored_source = root / 'mirrors' / f'mirror_{index:04d}.png'
                        flipped = cv2.flip(aggregate.read_image(snapshot), 1)
                        encoded, pixels = cv2.imencode('.png', flipped)
                        if not encoded:
                            raise ValueError('镜像图片生成失败')
                        pixels.tofile(str(mirrored_source))

                        def mirror_progress(event):
                            progress({**event, 'stage': 'mirror_' + event['stage'],
                                      'message': '镜像图 · ' + event['message']})

                        mirrored = aggregate.aggregate_image(mirrored_source, face=False, pose=False,
                                                             multimodal=False, progress=mirror_progress, mirror_selection=False)
                        mirrored_destination = root / 'mirror-results' / mirrored['result']
                        shutil.copyfile(aggregate.RESULT_DIR / mirrored['result'], mirrored_destination)
                        mirrored['source_url'] = f'{prefix}/mirrors/{quote(mirrored_source.name)}'
                        mirrored['result_url'] = f'{prefix}/mirror-results/{quote(mirrored_destination.name)}'
                        with aggregate_jobs._lock:
                            row['mirror_result'] = mirrored
                    with aggregate_jobs._lock:
                        row.update(status='completed', result=result)
                        aggregate_jobs.JOBS[job_id]['completed_images'] += 1
                except aggregate_jobs.Cancelled:
                    raise
                except Exception as error:
                    with aggregate_jobs._lock:
                        row.update(status='failed', error=str(error))
                        aggregate_jobs.JOBS[job_id]['failed_images'] += 1
                update(percent=round((index + 1) / len(sources) * 100, 1),
                       elapsed_seconds=round(time.time() - aggregate_jobs.JOBS[job_id]['created_at'], 1))
                persist(job_id)
            job = aggregate_jobs.get(job_id)
            update(status='completed' if job['completed_images'] else 'failed', percent=100,
                   message=f"批量处理结束，成功 {job['completed_images']} 张，失败 {job['failed_images']} 张")
        except aggregate_jobs.Cancelled:
            with aggregate_jobs._lock:
                job = aggregate_jobs.JOBS[job_id]
                job.update(status='cancelled', message='推理已停止，已完成结果可继续查看')
                for row in job['results']:
                    if row['status'] in ('waiting', 'running'):
                        row['status'] = 'cancelled'
                for step in job['stages']:
                    if step['status'] == 'running':
                        step['status'] = 'cancelled'
        except Exception as error:
            update(status='failed', error=str(error), message=str(error))
        finally:
            update(elapsed_seconds=round(time.time() - aggregate_jobs.JOBS[job_id]['created_at'], 1))
            try:
                persist(job_id)
            finally:
                aggregate_jobs._runner.release()

    try:
        persist(job_id)
        threading.Thread(target=work, name=f'aggregate-batch-{job_id[:8]}', daemon=True).start()
    except Exception:
        with aggregate_jobs._lock:
            aggregate_jobs.JOBS.pop(job_id, None)
        aggregate_jobs._runner.release()
        raise
    return aggregate_jobs.get(job_id)
