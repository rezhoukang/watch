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


def start_recognition(run_id: str, threshold: float = 0.40,
                      min_confidence: float = 0.5, min_size: int = 24,
                      min_sharpness: float = 50.0) -> str:
    """对批次中所有已检出的人脸做 1:N 识别（后台任务，结果写 recognition.json）。

    质量门槛均可调（不再是硬编码 0.8）：
    - min_confidence 默认 0.5，与 SCRFD 检出阈值一致，避免重复设卡；
    - min_size 是人脸在原始像素下的短边下限（默认 24）。
    """
    import face_recognition

    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError('识别阈值必须在0到1之间')
    if not math.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
        raise ValueError('人脸置信度门槛必须在0到1之间')
    if isinstance(min_size, bool) or not isinstance(min_size, int) or not 1 <= min_size <= 1000:
        raise ValueError('最小人脸像素必须在1到1000之间')
    if not math.isfinite(min_sharpness) or not 0.0 <= min_sharpness <= 10000:
        raise ValueError('清晰度门槛必须在0到10000之间')
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
        # 收集所有已检出的人脸：识别必须用 HC 裁剪图，因为 face 的 bbox/landmarks 都是相对它的坐标
        crops = []
        detection = batch.get('face_detection') or {}
        for crop in batch['crops']:
            per_crop = (crop.get('face_detection') or {})
            faces = per_crop.get('faces', []) if per_crop.get('status') == 'detected' else []
            if faces:
                crops.append({'crop_id': crop['crop_id'], 'raw_file': crop['raw_file'], 'faces': faces})
        if not crops:
            raise ValueError('当前批次没有人脸可识别，请先运行人脸检测')
        job_id = uuid.uuid4().hex
        record = {'job_id': job_id, 'run_id': run_id, 'status': 'queued',
                  'created_at': datetime.now(timezone.utc).isoformat(), 'message': '等待 1:N 识别…',
                  'config': {'model': 'w600k_r50.onnx', 'threshold': threshold,
                             'min_confidence': min_confidence, 'min_size': min_size,
                             'min_sharpness': min_sharpness,
                             'library_count': len(face_recognition.list_entries())},
                  'total': sum(len(c['faces']) for c in crops), 'progress': 0,
                  'matched': 0, 'unknown': 0, 'rejected': 0, 'reassigned': 0, 'conflicts': 0,
                  'results': {}, 'errors': []}
        _save_recognition(run_id, record)
        vision.JOBS[job_id] = {'id': job_id, 'type': 'head_face_recognition', 'run_id': run_id,
                               'status': 'running', 'total': record['total'], 'progress': 0, 'message': record['message']}
    threading.Thread(target=_run_recognition, args=(run_id, job_id, crops, record, threshold), daemon=True).start()
    return job_id


def _run_recognition(run_id: str, job_id: str, crops: list[dict], record: dict, threshold: float) -> None:
    import face_recognition

    job = vision.JOBS[job_id]
    # 质量门槛来自 config（可调），兼容旧记录缺字段的情况
    config = record.get('config') or {}
    min_confidence = config.get('min_confidence', 0.5)
    min_size = config.get('min_size', 24)
    min_sharpness = config.get('min_sharpness', 50)
    try:
        with _detection_lock:
            folder = head_cuts.run_dir(run_id)
            record['status'] = 'running'
            _save_recognition(run_id, record)
            # 第一阶段：逐张裁剪做质量筛选、提特征、取候选（结论留到第二阶段统一定）
            candidates_by_face: dict[str, list[dict]] = {}
            cancelled = False
            for crop in crops:
                if job.get('cancel'):
                    cancelled = True
                    break
                # 打开 HC 裁剪图：face 的 bbox/landmarks 都相对它，
                # 若改用单张人脸裁剪图（faces/…png）则坐标系不一致，会导致质量取样和对齐都错位。
                raw_file = crop.get('raw_file')
                source_path = (folder/raw_file).resolve() if raw_file else None
                source = cv2.imdecode(np.fromfile(str(source_path), dtype=np.uint8), cv2.IMREAD_COLOR) if source_path and source_path.is_file() else None
                per_face = []
                for face in crop['faces']:
                    if source is None:
                        per_face.append({'face_id': face['face_id'], 'status': 'error', 'message': '无法读取原始裁剪图'})
                        continue
                    try:
                        # 质量筛选与对齐都在 HC 坐标系下进行（bbox/landmarks 即原图像素尺度）。
                        landmarks = face.get('landmarks')
                        bbox = face.get('bbox', [0, 0, 0, 0])
                        reasons, quality = face_recognition.assess_quality(
                            source, {'bbox': bbox, 'landmarks': landmarks, 'confidence': face.get('confidence', 0.5)},
                            min_size=min_size, min_sharpness=min_sharpness, min_confidence=min_confidence)
                        # 人脸在原始像素下的短边尺寸：供人工判断可识别性（24px 只是下限，不等于可靠）
                        face_pixels = int(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
                        if reasons:
                            record['rejected'] += 1
                            per_face.append({'face_id': face['face_id'], 'status': 'rejected',
                                             'reasons': reasons, 'face_pixels': face_pixels,
                                             'quality': quality})
                        elif landmarks:
                            embedding = face_recognition.extract_embedding(source, landmarks)
                            candidates = face_recognition.match_top_k(embedding, k=face_recognition.CANDIDATE_K)
                            candidates_by_face[face['face_id']] = candidates
                            per_face.append({'face_id': face['face_id'], 'status': 'pending',
                                             'face_pixels': face_pixels, 'candidates': candidates})
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

            # 第二阶段：一对一消解——同一库内编号只给得分最高的那张脸，被抢的脸顺延到下一个候选
            resolution = face_recognition.resolve_one_to_one(candidates_by_face, threshold)
            for info in record['results'].values():
                for entry in info['faces']:
                    if entry.get('status') != 'pending':
                        continue
                    outcome = resolution.get(entry['face_id']) or {'assigned': None, 'skipped': []}
                    assigned = outcome['assigned']
                    best = (entry.get('candidates') or [None])[0]
                    entry['conflict'] = {'skipped': outcome['skipped']}
                    if assigned is not None:
                        entry['status'] = 'matched'
                        entry['match'] = {**assigned, 'matched': True}
                        record['matched'] += 1
                        if assigned['rank'] > 1:
                            record['reassigned'] += 1
                    else:
                        entry['status'] = 'unknown'
                        entry['match'] = {**(best or {}), 'matched': False,
                                          'reason': 'taken' if outcome['skipped'] else 'below_threshold'}
                        record['unknown'] += 1
                    if outcome['skipped']:
                        record['conflicts'] += 1

            if cancelled:
                record.update(status='cancelled', message='已取消人脸识别，已生成结果保留')
            else:
                record.update(status='done_with_errors' if record['errors'] else 'done',
                              message=(f'人脸识别完成：命中 {record["matched"]}，未识别 {record["unknown"]}，'
                                       f'质量门槛拒绝 {record.get("rejected", 0)}；'
                                       f'同一人被多张脸命中已消解 {record.get("conflicts", 0)} 处，'
                                       f'顺延到备选 {record.get("reassigned", 0)} 张'))
    except Exception as exc:
        record.update(status='failed', message=f'人脸识别失败：{exc}')
    finally:
        _save_recognition(run_id, record)
        job.update(status=record['status'], message=record['message'], progress=record['progress'])
