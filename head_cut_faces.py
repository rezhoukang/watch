"""SCRFD detection on saved head crops, preserving each parent HC identifier."""
import math
import threading
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image, ImageDraw

import head_cuts
import vision
from scrfd_detector import ScrfdDetector, MODEL_PATH, MODEL_SHA256

_detection_lock = threading.Lock()
_detector = None


def _get_detector() -> ScrfdDetector:
    global _detector
    if _detector is None:
        _detector = ScrfdDetector()
    return _detector


def detect_crop(folder, crop: dict, job_id: str, detector: ScrfdDetector, confidence: float) -> dict:
    path = (folder/crop['raw_file']).resolve()
    if not path.is_relative_to((folder/'crops').resolve()):
        raise ValueError('只能检测 crops/ 中的原始裁剪图')
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('无法读取原始裁剪图')
    predictions = detector.detect(image, threshold=confidence)
    height, width = image.shape[:2]
    faces = []
    annotated = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    drawing = ImageDraw.Draw(annotated)
    for prediction in sorted(predictions, key=lambda item: -item['confidence']):
        bbox = prediction['bbox']
        if not all(math.isfinite(v) for v in bbox) or prediction['confidence'] < confidence:
            continue
        left, top = max(0, math.floor(bbox[0])), max(0, math.floor(bbox[1]))
        right, bottom = min(width, math.ceil(bbox[2])), min(height, math.ceil(bbox[3]))
        if right <= left or bottom <= top:
            continue
        face_id = f'{crop["crop_id"]}_F{len(faces)+1:02d}'
        relative = f'faces/{job_id}/{face_id}.png'
        target = folder/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)).save(target)
        x, y = crop['crop_bbox'][:2]
        faces.append({'face_id': face_id, 'parent_crop_id': crop['crop_id'],
                      'confidence': round(prediction['confidence'], 4), 'bbox': [round(v, 2) for v in bbox],
                      'source_bbox': [round(bbox[0]+x, 2), round(bbox[1]+y, 2), round(bbox[2]+x, 2), round(bbox[3]+y, 2)],
                      'landmarks': [[round(v, 2) for v in p] for p in prediction['landmarks']],
                      'file': relative, 'width': right-left, 'height': bottom-top})
        drawing.rectangle((left, top, right-1, bottom-1), outline='#00bb66', width=1)
        for px, py in prediction['landmarks']:
            drawing.ellipse((px-1, py-1, px+1, py+1), fill='#ff4081')
    annotated_file = None
    if faces:
        annotated_file = f'face_annotations/{job_id}/{crop["crop_id"]}.png'
        (folder/annotated_file).parent.mkdir(parents=True, exist_ok=True)
        annotated.save(folder/annotated_file)
    return {'crop_id': crop['crop_id'], 'status': 'detected' if faces else 'no_face',
            'face_count': len(faces), 'faces': faces, 'annotated_file': annotated_file}


def _save(run_id: str, record: dict) -> None:
    with head_cuts._lock:
        head_cuts._write(head_cuts.run_dir(run_id)/'face_detection.json', record)


def start_detection(run_id: str, confidence: float = .5) -> str:
    if not math.isfinite(confidence) or not .01 <= confidence <= .99:
        raise ValueError('人脸检测置信度必须在0.01到0.99之间')
    with head_cuts._lock:
        batch = head_cuts.read_run(run_id)
        if batch['status'] in ('queued', 'running'):
            raise ValueError('请先等待头切割完成')
        if not batch['crops']:
            raise ValueError('当前批次没有可检测的原始裁剪图')
        previous = batch.get('face_detection', {})
        if previous.get('status') in ('queued', 'running'):
            raise ValueError('本批次已有运行中的人脸检测任务')
        if not MODEL_PATH.is_file():
            raise ValueError('SCRFD 检测权重不存在，请配置 weight/scrfd/det_10g.onnx')
        job_id = uuid.uuid4().hex
        record = {'job_id': job_id, 'run_id': run_id, 'status': 'queued',
                  'created_at': datetime.now(timezone.utc).isoformat(), 'message': '等待 SCRFD 检测…',
                  'config': {'model': 'SCRFD-10G', 'confidence': confidence, 'input_size': 640, 'sha256': MODEL_SHA256},
                  'total': len(batch['crops']), 'progress': 0, 'heads_with_faces': 0,
                  'face_count': 0, 'results': {}, 'errors': []}
        _save(run_id, record)
        vision.JOBS[job_id] = {'id': job_id, 'type': 'head_face_detection', 'run_id': run_id,
                               'status': 'running', 'total': record['total'], 'progress': 0, 'message': record['message']}
    threading.Thread(target=_run_detection, args=(run_id, job_id, batch['crops'], record), daemon=True).start()
    return job_id


def _run_detection(run_id: str, job_id: str, crops: list[dict], record: dict) -> None:
    job = vision.JOBS[job_id]
    try:
        with _detection_lock:
            detector = _get_detector()
            folder = head_cuts.run_dir(run_id)
            record['status'] = 'running'
            _save(run_id, record)
            for index, crop in enumerate(crops, 1):
                if job.get('cancel'):
                    record.update(status='cancelled', message='已取消人脸检测，已生成结果保留')
                    break
                try:
                    result = detect_crop(folder, crop, job_id, detector, record['config']['confidence'])
                    record['heads_with_faces'] += int(result['status'] == 'detected')
                    record['face_count'] += result['face_count']
                except Exception as exc:
                    result = {'crop_id': crop['crop_id'], 'status': 'error', 'faces': [], 'face_count': 0, 'message': str(exc)}
                    record['errors'].append({'crop_id': crop['crop_id'], 'message': str(exc)})
                record['results'][crop['crop_id']] = result
                record.update(progress=index, message=f'SCRFD 检测 {index}/{len(crops)}：{crop["crop_id"]}')
                job.update(progress=index, message=record['message'])
                if index % 5 == 0:
                    _save(run_id, record)
            else:
                record.update(status='done_with_errors' if record['errors'] else 'done',
                              message=f'人脸检测完成：{record["heads_with_faces"]}/{len(crops)} 个 Head 检出人脸，共 {record["face_count"]} 张人脸')
    except Exception as exc:
        record.update(status='failed', message=f'SCRFD 人脸检测失败：{exc}')
    finally:
        _save(run_id, record)
        job.update(status=record['status'], message=record['message'], progress=record['progress'])


# ---------------------------------------------------------------- 1:N 人脸识别

def _save_recognition(run_id: str, record: dict) -> None:
    with head_cuts._lock:
        head_cuts._write(head_cuts.run_dir(run_id)/'recognition.json', record)


def start_recognition(run_id: str, threshold: float = 0.40) -> str:
    """对批次中所有已检出的人脸做 1:N 识别（后台任务，结果写 recognition.json）。"""
    import face_recognition

    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError('识别阈值必须在0到1之间')
    with head_cuts._lock:
        batch = head_cuts.read_run(run_id)
        if batch['status'] in ('queued', 'running'):
            raise ValueError('请先等待头切割完成')
        previous = batch.get('recognition', {})
        if previous.get('status') in ('queued', 'running'):
            raise ValueError('本批次已有运行中的人脸识别任务')
        if not face_recognition.MODEL_PATH_RECOGNITION.is_file():
            raise ValueError('识别模型不存在，请配置 weight/scrfd/w600k_r50.onnx')
        if not face_recognition.list_entries():
            raise ValueError('人脸库为空，请先导入学生花名册或手动添加')
        # 收集所有已检出的人脸（crop_id -> 人脸列表）
        crops = []
        detection = batch.get('face_detection') or {}
        for crop in batch['crops']:
            per_crop = (crop.get('face_detection') or {})
            faces = per_crop.get('faces', []) if per_crop.get('status') == 'detected' else []
            if faces:
                crops.append({'crop_id': crop['crop_id'], 'faces': faces})
        if not crops:
            raise ValueError('当前批次没有人脸可识别，请先运行人脸检测')
        job_id = uuid.uuid4().hex
        record = {'job_id': job_id, 'run_id': run_id, 'status': 'queued',
                  'created_at': datetime.now(timezone.utc).isoformat(), 'message': '等待 1:N 识别…',
                  'config': {'model': 'w600k_r50.onnx', 'threshold': threshold, 'library_count': len(face_recognition.list_entries())},
                  'total': sum(len(c['faces']) for c in crops), 'progress': 0,
                  'matched': 0, 'unknown': 0, 'results': {}, 'errors': []}
        _save_recognition(run_id, record)
        vision.JOBS[job_id] = {'id': job_id, 'type': 'head_face_recognition', 'run_id': run_id,
                               'status': 'running', 'total': record['total'], 'progress': 0, 'message': record['message']}
    threading.Thread(target=_run_recognition, args=(run_id, job_id, crops, record, threshold), daemon=True).start()
    return job_id


def _run_recognition(run_id: str, job_id: str, crops: list[dict], record: dict, threshold: float) -> None:
    import face_recognition

    job = vision.JOBS[job_id]
    try:
        with _detection_lock:
            folder = head_cuts.run_dir(run_id)
            record['status'] = 'running'
            _save_recognition(run_id, record)
            for crop in crops:
                if job.get('cancel'):
                    record.update(status='cancelled', message='已取消人脸识别，已生成结果保留')
                    break
                per_face = []
                for face in crop['faces']:
                    path = (folder/face['file']).resolve()
                    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        per_face.append({'face_id': face['face_id'], 'status': 'error', 'message': '无法读取人脸裁剪'})
                        continue
                    try:
                        # 人脸裁剪图通常很小，直接复用 head_cuts 已保存的关键点做对齐提特征，
                        # 避免二次检测在小图上失败；质量筛选仍是查询端防误报关卡。
                        landmarks = face.get('landmarks')
                        reasons, quality = face_recognition.assess_quality(image, {'bbox': face.get('bbox', [0, 0, 0, 0]), 'landmarks': landmarks, 'confidence': face.get('confidence', 0.5)})
                        if reasons:
                            per_face.append({'face_id': face['face_id'], 'status': 'rejected',
                                             'reasons': reasons, 'faces': []})
                        elif landmarks:
                            embedding = face_recognition.extract_embedding(image, landmarks)
                            match_result = face_recognition.match(embedding, threshold)
                            per_face.append({
                                'face_id': face['face_id'],
                                'status': 'matched' if match_result['matched'] else 'unknown',
                                'match': match_result,
                                'candidates': face_recognition.match_top_k(embedding, k=3),
                                'faces': [{'status': 'matched' if match_result['matched'] else 'unknown', 'match': match_result}] if match_result['matched'] else [{'status': 'unknown', 'match': match_result}],
                            })
                            if match_result['matched']:
                                record['matched'] += 1
                            else:
                                record['unknown'] += 1
                        else:
                            per_face.append({'face_id': face['face_id'], 'status': 'error', 'message': '缺少人脸关键点'})
                    except Exception as exc:  # noqa: BLE001
                        per_face.append({'face_id': face['face_id'], 'status': 'error', 'message': str(exc)})
                        record['errors'].append({'crop_id': crop['crop_id'], 'face_id': face['face_id'], 'message': str(exc)})
                    record['progress'] += 1
                    job.update(progress=record['progress'],
                               message=f'识别 {record["progress"]}/{record["total"]}：{face["face_id"]}')
                record['results'][crop['crop_id']] = {'crop_id': crop['crop_id'], 'status': 'done', 'faces': per_face}
                if record['progress'] % 5 == 0:
                    _save_recognition(run_id, record)
            else:
                record.update(status='done_with_errors' if record['errors'] else 'done',
                              message=f'人脸识别完成：命中 {record["matched"]}，未识别 {record["unknown"]}')
    except Exception as exc:
        record.update(status='failed', message=f'人脸识别失败：{exc}')
    finally:
        _save_recognition(run_id, record)
        job.update(status=record['status'], message=record['message'], progress=record['progress'])
