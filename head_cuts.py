"""Persistent batch crops of every box returned by the target-detection fusion."""
import json
import math
import re
import shutil
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

import head_fusion
import vision
from project_paths import BASE_DIR

HEAD_CUT_DIR = BASE_DIR / 'output' / 'head_cuts'
_lock = threading.RLock()
_inference_lock = threading.Lock()


def run_dir(run_id: str) -> Path:
    if not re.fullmatch(r'[0-9a-f]{32}', run_id):
        raise ValueError('非法头切割批次编号')
    folder = HEAD_CUT_DIR / run_id
    if folder.is_symlink() or folder.resolve().parent != HEAD_CUT_DIR.resolve():
        raise ValueError('非法头切割路径')
    return folder


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_run(run_id: str) -> dict:
    with _lock:
        folder = run_dir(run_id)
        record = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if record['status'] in ('running', 'queued') and run_id not in vision.JOBS:
            record['status'] = 'interrupted'
            record['message'] = '服务重启，批次未完成；已生成的裁剪图仍可查看，请重新推理未完成图片'
        marks = json.loads((folder/'marks.json').read_text(encoding='utf-8')) if (folder/'marks.json').is_file() else {}
        for item in record['crops']:
            item['mark'] = marks.get(item['crop_id'], '')
        face_path = folder/'face_detection.json'
        if face_path.is_file():
            faces = json.loads(face_path.read_text(encoding='utf-8'))
            if faces['status'] in ('running', 'queued') and faces['job_id'] not in vision.JOBS:
                faces.update(status='interrupted', message='服务重启，人脸检测中断；已完成结果保留，可重新检测')
            record['face_detection'] = {k: v for k, v in faces.items() if k != 'results'}
            for item in record['crops']:
                item['face_detection'] = faces['results'].get(item['crop_id'], {'status': 'pending', 'faces': [], 'face_count': 0})
        recognition_path = folder/'recognition.json'
        if recognition_path.is_file():
            recognized = json.loads(recognition_path.read_text(encoding='utf-8'))
            if recognized['status'] in ('running', 'queued') and recognized['job_id'] not in vision.JOBS:
                recognized.update(status='interrupted', message='服务重启，识别中断；已完成结果保留，可重新识别')
            record['recognition'] = {k: v for k, v in recognized.items() if k != 'results'}
            for item in record['crops']:
                per_crop = recognized['results'].get(item['crop_id'], {'status': 'pending'})
                names = []
                for face in per_crop.get('faces', []):
                    first = (face.get('faces') or [{}])[0]
                    if first.get('status') == 'matched' and first.get('match', {}).get('name'):
                        names.append(first['match']['name'])
                item['recognition'] = dict(per_crop, name='、'.join(names))
        return record


def list_runs() -> list[dict]:
    runs = []
    for path in HEAD_CUT_DIR.glob('*/manifest.json'):
        record = read_run(path.parent.name)
        runs.append({k: record[k] for k in ('run_id', 'created_at', 'status', 'total', 'progress', 'crop_count', 'message')})
    return sorted(runs, key=lambda r: r['created_at'], reverse=True)


def _update(run_id: str, **changes) -> None:
    with _lock:
        path = run_dir(run_id)/'manifest.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        record.update(changes)
        _write(path, record)


def save_marks(run_id: str, marks: dict) -> None:
    if not isinstance(marks, dict) or any(not isinstance(v, str) or len(v) > 80 for v in marks.values()):
        raise ValueError('标记必须是最多80字的文本')
    with _lock:
        record = read_run(run_id)
        known = {v['crop_id'] for v in record['crops']}
        if any(k not in known for k in marks):
            raise ValueError('标记中包含不存在的裁剪编号')
        path = run_dir(run_id)/'marks.json'
        previous = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        previous.update({k: v.strip() for k, v in marks.items()})
        _write(path, previous)


def crop_image(source: Path, source_name: str, image_index: int, folder: Path, parameters: dict) -> dict:
    prediction = head_fusion.detect_image(source, rtdetr_base=parameters['rtdetr_base'],
        merge_iou=parameters['merge_iou'], merge_confidence=parameters['merge_confidence'])
    with Image.open(source) as opened:
        image = opened.convert('RGB')
    annotated = image.copy()
    drawing = ImageDraw.Draw(annotated)
    crops, rejected = [], []
    for index, det in enumerate(prediction['detections'], 1):
        crop_id = f'HC{image_index:03d}-{index:03d}'
        bbox = det.get('bbox', [])
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
            rejected.append({'crop_id': crop_id, 'reason': '框坐标无效'})
            continue
        x1, y1, x2, y2 = bbox
        left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
        right, bottom = min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2))
        if right <= left or bottom <= top:
            rejected.append({'crop_id': crop_id, 'reason': '框不在图片范围内'})
            continue
        crop = image.crop((left, top, right, bottom))
        raw_path, marked_path = f'crops/{crop_id}.png', f'marked/{crop_id}.png'
        crop.save(folder/raw_path)
        label_box = drawing.textbbox((0, 0), crop_id)
        label_width = label_box[2]-label_box[0]+12
        marked = Image.new('RGB', (max(crop.width, label_width), crop.height+24), 'white')
        marked.paste(crop, ((marked.width-crop.width)//2, 0))
        ImageDraw.Draw(marked).text((6, crop.height+5), crop_id, fill='#124a7b')
        marked.save(folder/marked_path)
        drawing.rectangle((left, top, right-1, bottom-1), outline='#00bb66', width=2)
        drawing.text((left, max(0, top-12)), crop_id, fill='#0055aa', stroke_width=1, stroke_fill='white')
        crops.append({'crop_id': crop_id, 'source': source_name, 'image_index': image_index,
                      'class_name': det['class_name'], 'confidence': det['confidence'],
                      'bbox': list(bbox), 'crop_bbox': [left, top, right, bottom],
                      'width': crop.width, 'height': crop.height, 'source_model': det.get('source_model'),
                      'raw_file': raw_path, 'marked_file': marked_path})
    annotated_path = f'annotated/{image_index:03d}.png'
    annotated.save(folder/annotated_path)
    return {'source': source_name, 'image_index': image_index, 'crop_count': len(crops),
            'detection_count': len(prediction['detections']), 'annotated_file': annotated_path,
            'crops': crops, 'rejected': rejected}


def start_batch(sources: list[Path], rtdetr_base=False, merge_iou=.1, merge_confidence=.55) -> str:
    parameters = head_fusion.parameters(rtdetr_base, merge_iou, merge_confidence)
    run_id = uuid.uuid4().hex
    folder = run_dir(run_id)
    for name in ('sources', 'crops', 'marked', 'annotated'):
        (folder/name).mkdir(parents=True, exist_ok=True)
    inputs = []
    for index, path in enumerate(sources, 1):
        snapshot = f'sources/{index:03d}{path.suffix.lower()}'
        shutil.copyfile(path, folder/snapshot)
        inputs.append({'filename': path.name, 'snapshot_file': snapshot})
    record = {'run_id': run_id, 'created_at': datetime.now(timezone.utc).isoformat(), 'parameters': parameters,
              'status': 'queued', 'message': '等待目标检测…', 'total': len(inputs), 'progress': 0,
              'crop_count': 0, 'inputs': inputs, 'images': [], 'crops': [], 'errors': []}
    _write(folder/'manifest.json', record)
    vision.JOBS[run_id] = {'id': run_id, 'type': 'head_cut', 'status': 'running', 'message': record['message'],
                           'total': len(inputs), 'progress': 0, 'run_id': run_id}
    threading.Thread(target=_run_batch, args=(run_id,), daemon=True).start()
    return run_id


def _run_batch(run_id: str) -> None:
    job = vision.JOBS[run_id]
    try:
        with _inference_lock:
            folder = run_dir(run_id)
            record = read_run(run_id)
            images, crops, errors = [], [], []
            for index, item in enumerate(record['inputs'], 1):
                if job.get('cancel'):
                    job.update(status='cancelled', message='已取消；已生成的裁剪图仍保留')
                    _update(run_id, status=job['status'], message=job['message'])
                    return
                message = f'正在处理 {index}/{record["total"]}：{item["filename"]}'
                job['message'] = message
                _update(run_id, status='running', message=message)
                try:
                    result = crop_image(folder/item['snapshot_file'], item['filename'], index, folder, record['parameters'])
                    images.append({k: v for k, v in result.items() if k != 'crops'})
                    crops.extend(result['crops'])
                except Exception as exc:
                    errors.append({'source': item['filename'], 'message': str(exc)})
                job['progress'] = index
                _update(run_id, progress=index, images=images, crops=crops, errors=errors, crop_count=len(crops))
            status = 'done_with_errors' if errors else 'done'
            message = f'处理完成：{len(record["inputs"])} 张图片，{len(crops)} 个裁剪；失败 {len(errors)} 张'
            job.update(status=status, message=message, crop_count=len(crops))
            _update(run_id, status=status, message=message)
    except Exception as exc:
        job.update(status='failed', message=f'头切割失败：{exc}')
        _update(run_id, status='failed', message=job['message'])


def build_archive(run_id: str) -> Path:
    with _lock:
        record = read_run(run_id)
        if record['status'] in ('running', 'queued') or record.get('face_detection', {}).get('status') in ('running', 'queued'):
            raise ValueError('请等待批次完成后再下载')
        folder = run_dir(run_id)
        archive = HEAD_CUT_DIR/f'{run_id}.zip'
        newest = max(p.stat().st_mtime_ns for p in (folder/'manifest.json', folder/'marks.json', folder/'face_detection.json') if p.is_file())
        if archive.is_file() and archive.stat().st_mtime_ns >= newest:
            return archive
        temporary = archive.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for file in folder.rglob('*'):
                if file.is_file() and not file.name.endswith('.tmp'):
                    bundle.write(file, file.relative_to(folder))
        temporary.replace(archive)
        return archive
