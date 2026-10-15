"""Per-image anonymous Head/face records and an offline manual review report."""
import hashlib
import html
import json
import math
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import head_fusion
from inference_history import box_iou
from project_paths import BASE_DIR, UPLOAD_DIR, IMAGE_EXTENSIONS
from scrfd_detector import ScrfdDetector, assess_quality, align_face, MODEL_SHA256, MODEL_SOURCE


def select_classroom_images(directory: Path = UPLOAD_DIR) -> list[tuple[str, Path]]:
    images = []
    for path in directory.iterdir():
        match = re.fullmatch(r'([ab])教室\s*\((\d+)\)', path.stem, re.I)
        if match and path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append((f'{match[1].upper()}{int(match[2]):02d}', path))
    images.sort(key=lambda item: item[0])
    if len({v[0] for v in images}) != len(images):
        raise ValueError('同一教室图片编号重复，请先区分文件名')
    return images


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f'无法编码图片：{path}')
    encoded.tofile(str(path))


def clip_bbox(box, width, height, padding=0.0) -> list[int]:
    x1, y1, x2, y2 = box
    dx, dy = (x2-x1)*padding, (y2-y1)*padding
    return [max(0, min(width, math.floor(x1-dx))), max(0, min(height, math.floor(y1-dy))),
            max(0, min(width, math.ceil(x2+dx))), max(0, min(height, math.ceil(y2+dy)))]


def process_image(source: Path, image_id: str, output: Path, detector: ScrfdDetector) -> dict:
    image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法读取图片：{source.name}')
    height, width = image.shape[:2]
    prediction = head_fusion.detect_image(source)
    heads = [v for v in prediction['detections'] if v['class_name'].lower() == 'head']
    heads.sort(key=lambda v: (v['bbox'][1], v['bbox'][0]))
    records, seen_faces = [], []
    annotated = image.copy()
    for number, head in enumerate(heads, 1):
        anonymous_id = f'{image_id}-{number:03d}'
        head_bbox = clip_bbox(head['bbox'], width, height)
        crop_bbox = clip_bbox(head['bbox'], width, height, .15)
        x1, y1, x2, y2 = crop_bbox
        record = {'anonymous_id': anonymous_id, 'image_id': image_id, 'source': source.name,
                  'head_bbox': head_bbox, 'head_roi_bbox': crop_bbox, 'head_confidence': head['confidence'],
                  'head_source_model': head.get('source_model'), 'status': 'skipped', 'reasons': [],
                  'manual_name': '', 'head_image': f'{image_id}/heads/{anonymous_id}.jpg'}
        records.append(record)
        if x2 <= x1 or y2 <= y1:
            record['reasons'] = ['Head 框无效']
            record['head_image'] = None
            continue
        crop = image[y1:y2, x1:x2].copy()
        save_image(output / record['head_image'], crop)
        candidates = []
        for face in detector.detect(crop):
            a, b, c, d = face['bbox']
            cx, cy = (a+c)/2+x1, (b+d)/2+y1
            if not (head_bbox[0] <= cx <= head_bbox[2] and head_bbox[1] <= cy <= head_bbox[3]):
                continue
            face = {**face, 'bbox': [a+x1, b+y1, c+x1, d+y1],
                    'landmarks': [[x+x1, y+y1] for x, y in face['landmarks']]}
            candidates.append(face)
        candidates.sort(key=lambda f: -f['confidence'])
        if not candidates:
            record['reasons'] = ['未检测到可靠人脸']
        else:
            face = candidates[0]
            reasons, metrics = assess_quality(image, face)
            if sum(f['confidence'] >= .8 for f in candidates) > 1:
                reasons.append('同一 Head 内有多个人脸候选')
            if any(box_iou(face['bbox'], prior) > .4 for prior in seen_faces):
                reasons.append('与本图已记录人脸重复')
            else:
                seen_faces.append(face['bbox'])
            record.update({'face_bbox': [round(v, 2) for v in face['bbox']],
                           'landmarks': [[round(v, 2) for v in p] for p in face['landmarks']],
                           'face_confidence': round(face['confidence'], 4), 'quality': metrics, 'reasons': reasons})
            a, b, c, d = clip_bbox(face['bbox'], width, height)
            if c > a and d > b:
                record['face_image'] = f'{image_id}/faces/{anonymous_id}.jpg'
                save_image(output / record['face_image'], image[b:d, a:c])
            if not reasons:
                try:
                    aligned = align_face(image, face['landmarks'])
                    record['aligned_image'] = f'{image_id}/aligned/{anonymous_id}.png'
                    save_image(output / record['aligned_image'], aligned)
                    record['status'] = 'accepted'
                except ValueError:
                    record['reasons'] = ['人脸对齐失败']
        color = (0, 180, 0) if record['status'] == 'accepted' else (0, 110, 255)
        a, b, c, d = head_bbox
        cv2.rectangle(annotated, (a, b), (c, d), color, 1)
        cv2.putText(annotated, anonymous_id, (a, max(10, b-3)), cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1, cv2.LINE_AA)
    annotated_path = f'{image_id}/annotated.jpg'
    save_image(output / annotated_path, annotated)
    return {'image_id': image_id, 'source': source.name, 'width': width, 'height': height,
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'annotated_image': annotated_path, 'head_count': len(heads),
            'accepted_count': sum(v['status'] == 'accepted' for v in records), 'records': records}


def write_review(output: Path, manifest: dict) -> None:
    from urllib.parse import quote

    sections = []
    for image in manifest['images']:
        cards = []
        for row in image['records']:
            head = row.get('head_image')
            face = row.get('aligned_image') or row.get('face_image')
            pictures = (f'<img src="{quote(head)}" alt="Head裁剪">' if head else '')
            if face:
                pictures += f'<img src="{quote(face)}" alt="人脸裁剪">'
            reasons = '；'.join(row['reasons']) if row['reasons'] else '通过初步质量筛选'
            cards.append(f'<article class="card" data-id="{row["anonymous_id"]}" data-state="{row["status"]}"><b>{row["anonymous_id"]}</b><div class="photos">{pictures}</div><p>{html.escape(reasons)}</p><label>人工姓名 <input data-name="{row["anonymous_id"]}" placeholder="可留空" maxlength="80"></label></article>')
        sections.append(f'<section><h2>{html.escape(image["source"])} · Head {image["head_count"]} / 通过 {image["accepted_count"]}</h2><details><summary>查看本图匿名编号位置</summary><a href="{quote(image["annotated_image"])}" target="_blank"><img class="scene" src="{quote(image["annotated_image"])}" alt="匿名编号总览"></a></details><div class="grid">{"".join(cards)}</div></section>')
    template = (BASE_DIR / 'scripts/anonymous_review.html').read_text(encoding='utf-8')
    script_data = json.dumps({'run_id': output.name, 'images': [{'image_id': i['image_id'], 'source': i['source'],
        'records': [{'anonymous_id': r['anonymous_id'], 'status': r['status']} for r in i['records']]} for i in manifest['images']]}, ensure_ascii=True)
    page = template.replace('<!--SUMMARY-->', f'共 {manifest["summary"]["images"]} 张图、{manifest["summary"]["heads"]} 个 Head；通过 {manifest["summary"]["accepted"]}，跳过 {manifest["summary"]["skipped"]}。')
    page = page.replace('<!--CARDS-->', ''.join(sections)).replace('/*REPORT_DATA*/', script_data)
    (output / 'index.html').write_text(page, encoding='utf-8')


def run_batch(output: Path | None = None) -> Path:
    images = select_classroom_images()
    if len(images) != 10:
        raise ValueError(f'预期10张A/B教室图片，当前找到{len(images)}张；请核对待推理图片')
    output = output or BASE_DIR / 'output/anonymous_faces' / datetime.now().strftime('ab_classrooms_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    detector = ScrfdDetector()
    manifest = {'run_id': output.name, 'created_at': datetime.now().astimezone().isoformat(),
                'numbering': 'per_image_detection_record', 'head_parameters': head_fusion.parameters(),
                'face_detector': {'name': 'SCRFD-10G', 'sha256': MODEL_SHA256, 'source': MODEL_SOURCE,
                                  'input_size': 640, 'candidate_confidence': .5},
                'quality_rules': {'face_confidence': .8, 'minimum_original_face_size': 24,
                                  'minimum_laplacian_variance': 50, 'maximum_side_proxy': .45,
                                  'maximum_roll_degrees': 30}, 'images': []}
    for image_id, source in images:
        print(json.dumps({'stage': 'processing', 'image_id': image_id, 'source': source.name}), flush=True)
        result = process_image(source, image_id, output, detector)
        manifest['images'].append(result)
        print(json.dumps({'stage': 'finished', 'image_id': image_id, 'heads': result['head_count'], 'accepted': result['accepted_count']}), flush=True)
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    records = [r for i in manifest['images'] for r in i['records']]
    manifest['summary'] = {'images': len(images), 'heads': len(records),
                           'accepted': sum(r['status'] == 'accepted' for r in records),
                           'skipped': sum(r['status'] == 'skipped' for r in records),
                           'skip_reasons': dict(Counter(reason for r in records for reason in r['reasons']))}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'manual_names_template.json').write_text(json.dumps({'run_id': output.name,
        'labels': [{'anonymous_id': r['anonymous_id'], 'source': r['source'], 'manual_name': ''} for r in records]}, ensure_ascii=False, indent=2), encoding='utf-8')
    write_review(output, manifest)
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for file in output.rglob('*'):
            if file.is_file():
                bundle.write(file, Path(output.name) / file.relative_to(output))
    print(json.dumps({'stage': 'complete', 'output': str(output), 'summary': manifest['summary']}), flush=True)
    return output
