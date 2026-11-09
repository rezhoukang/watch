"""OpenCV sampling and persistent video target/action aggregation."""
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import aggregate
import head_fusion
from action_recognition import decoding, head_only, jobs, prompts, store, two_stage, vlm
from action_recognition.regions import Expansion

KIND = 'video_aggregate'


def read(run_id: str) -> dict:
    record = store.read(run_id)
    if record.get('source_kind') != KIND:
        raise FileNotFoundError('视频聚合批次不存在')
    return record


def targets(record: dict, image_index: int) -> list[dict]:
    return [{'crop_id': crop['crop_id'], 'bbox': crop['head_bbox'], 'class_name': 'head',
             'confidence': crop['head_confidence'], 'status': crop['status'],
             'action': crop.get('action'), 'error': crop.get('error')}
            for crop in record['crops'] if crop['image_index'] == image_index]


def describe(run_id: str) -> dict:
    record = read(run_id)
    frames = []
    groups = {}
    detected = {info['image_index'] for info in record['images']}
    for crop in record['crops']:
        groups.setdefault(crop['image_index'], []).append(crop)
    for frame in record['video']['frames']:
        items = groups.get(frame['image_index'], [])
        counts = Counter((c.get('action') or {}).get('state') or ('分析失败' if c['status'] == 'failed' else '待分析') for c in items)
        frames.append({**frame, 'count': len(items), 'counts': dict(counts), 'detected': frame['image_index'] in detected,
                       'provisional': any((c.get('action') or {}).get('provisional') for c in items)})
    active = jobs.active()
    status = record['status']
    if status in ('extracting', 'detecting', 'analyzing', 'rendering') and (not active or active['run_id'] != run_id):
        status = 'interrupted'
    return {'run_id': run_id, 'created_at': record['created_at'], 'status': status,
            'source': record['filenames'][0], 'video': {**record['video'], 'frames': frames},
            'crop_count': record['crop_count'], 'errors': record['errors'],
            'video_url': f'/api/video-aggregate/runs/{run_id}/video'}


def history() -> list[dict]:
    result = []
    for item in store.list_runs():
        try:
            record = read(item['run_id'])
            result.append({'run_id': item['run_id'], 'created_at': item['created_at'],
                           'source': record['filenames'][0], 'status': record['status'],
                           'interval_seconds': record['video']['interval_seconds'],
                           'frame_count': len(record['video']['frames'])})
        except (OSError, ValueError, KeyError):
            continue
    return result


def extract(record: dict, event, job_id: str) -> None:
    root = store.folder(record['run_id'])
    capture = cv2.VideoCapture(str(root / 'source.mp4'))
    try:
        if not capture.isOpened():
            raise ValueError('OpenCV 无法打开 MP4，请检查文件是否完整')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('无法读取有效视频帧率')
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = int(frame_count) if math.isfinite(frame_count) and frame_count > 0 else 0
        stride = max(1, round(record['video']['interval_seconds'] * fps))
        record['video'].update(fps=fps, source_frames=frame_count, stride_frames=stride,
                               actual_interval_seconds=stride / fps, duration_seconds=frame_count / fps)
        (root / 'sources').mkdir(exist_ok=True)
        index = 0
        while not event.is_set() and capture.grab():
            if index % stride == 0:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    raise ValueError(f'读取视频第 {index} 帧失败')
                number = len(record['video']['frames']) + 1
                filename = f'sources/{number:03d}.png'
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(root / filename)
                record['video']['frames'].append({'image_index': number, 'frame_index': index,
                                                  'time_seconds': round(index / fps, 4), 'source_file': filename})
                store.write(record)
                jobs.update(job_id, message=f'OpenCV 抽帧：已提取 {number} 张 · {index / fps:.2f}s',
                            frame_current=number, stage='extracting')
            index += 1
        if not record['video']['frames'] and not event.is_set():
            raise ValueError('视频中没有可读取的画面')
        if not event.is_set() and frame_count and index < frame_count - 1:
            raise ValueError(f'视频提前结束，读取到 {index}/{frame_count} 帧')
    finally:
        capture.release()


def start(source: Path, interval_seconds: float) -> dict:
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError('抽帧间隔必须大于 0 秒')
    if not jobs._worker.acquire(blocking=False):
        raise RuntimeError('已有动作或多模态任务正在运行，请等待完成')
    try:
        ready = vlm.status(vlm.DEFAULT_MODEL, '4bit')
        if not ready['available']:
            raise ValueError(ready['message'])
        profile = head_only.profile()
        context = two_stage.context_profile(prompts.load())
        record = store.create([source.name], Expansion())
        record.update(source_kind=KIND, status='extracting',
                      video={'interval_seconds': interval_seconds, 'frames': []})
        store.write(record)

        def work(job_id, event):
            try:
                shutil.copyfile(source, store.folder(record['run_id']) / 'source.mp4')
                extract(record, event, job_id)
                vlm.unload()
                record['status'] = 'detecting'
                record['head_parameters'] = head_fusion.parameters()
                root = store.folder(record['run_id'])
                for frame in record['video']['frames']:
                    if event.is_set():
                        break
                    number = frame['image_index']
                    jobs.update(job_id, stage='detecting', frame_current=number,
                                message=f"目标检测 {number}/{len(record['video']['frames'])} · {frame['time_seconds']:.2f}s")
                    path = root / frame['source_file']
                    prediction = head_fusion.detect_image(path)
                    with Image.open(path) as image:
                        store.append_image(record, f'{source.stem}_{frame["frame_index"]:08d}.png', number,
                                           image, prediction['detections'], Expansion())
                    store.write(record)
                if not event.is_set():
                    record['status'] = 'analyzing'
                    profile_id = prompts.identifier(profile)
                    record['prompt_profiles'] = {
                        prompts.identifier(cfg): {'config': cfg, 'compiled_prompt': prompts.compile_prompt(cfg),
                                                  'prompt_version': prompts.VERSION} for cfg in (profile, context)}
                    session = {'session_id': job_id, 'started_at': store.now(), 'status': 'running',
                               'model': vlm.MODELS[vlm.DEFAULT_MODEL], 'quantization': '4bit',
                               'decoding': decoding.describe('A'),
                               'prompt_config_id': profile_id, 'context_prompt_config_id': prompts.identifier(context),
                               'context_profile': context, 'head_only': True, 'two_stage': True,
                               'force': True, 'total': len(record['crops']), 'results': []}
                    record['analysis_sessions'] = [session]
                    jobs.update(job_id, total=len(record['crops']), progress=0)
                    started = time.perf_counter()
                    jobs._analyze_items(job_id, event, record, record['crops'], vlm.DEFAULT_MODEL,
                                        '4bit', True, profile, profile_id, 'A', session)
                    session.update(status='cancelled' if event.is_set() else 'completed',
                                   finished_at=store.now(), elapsed_seconds=round(time.perf_counter() - started, 3))
                if not event.is_set():
                    record['status'] = 'rendering'
                    (root / 'results').mkdir(exist_ok=True)
                    for frame in record['video']['frames']:
                        if event.is_set():
                            break
                        jobs.update(job_id, stage='rendering', message=f"生成采样结果图 {frame['image_index']}/{len(record['video']['frames'])}")
                        with Image.open(root / frame['source_file']) as image:
                            source_image = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
                        filename = f"results/{frame['image_index']:06d}.png"
                        aggregate.render(source_image, targets(record, frame['image_index'])).save(root / filename)
                        frame['result_file'] = filename
                record['status'] = 'cancelled' if event.is_set() else 'completed'
            except Exception as error:
                record['status'] = 'failed'
                record['errors'].append({'source': source.name, 'message': str(error)})
                for session in record.get('analysis_sessions', []):
                    if session['status'] == 'running':
                        session.update(status='failed', finished_at=store.now())
                raise
            finally:
                store.write(record)

        return jobs._launch(KIND, record['run_id'], 0, work)
    except Exception:
        jobs._worker.release()
        raise
