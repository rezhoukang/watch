"""Saved per-image aggregate results, backed by completed batch snapshots."""
import base64
import copy
import io
import json
import mimetypes
import re
import threading
import zipfile
from pathlib import Path

import aggregate
import aggregate_batch

_lock = threading.RLock()


def _deleted() -> set[str]:
    path = aggregate.RESULT_DIR / 'history-deleted.json'
    return set(json.loads(path.read_text(encoding='utf-8'))) if path.exists() else set()


def _read(record_id: str) -> dict:
    if not re.fullmatch(r'[a-f0-9]{32}-[0-9]+', record_id):
        raise FileNotFoundError('聚合记录不存在')
    batch_id, index = record_id.split('-')
    batch = aggregate_batch.saved(batch_id)
    try:
        row = batch['results'][int(index)]
        if row['status'] != 'completed':
            raise FileNotFoundError('聚合记录尚未完成')
        result = copy.deepcopy(row['result'])
    except (IndexError, KeyError):
        raise FileNotFoundError('聚合记录不存在') from None
    return {'id': record_id, 'source': row['source'], 'created_at': result.get('generated_at', ''),
            'count': result.get('count', 0), 'result': result}


def get(record_id: str) -> dict:
    with _lock:
        if record_id in _deleted():
            raise FileNotFoundError('聚合记录已删除')
        return _read(record_id)


def listing() -> list[dict]:
    with _lock:
        deleted = _deleted()
        records = []
        for path in (aggregate.RESULT_DIR / 'batches').glob('*/record.json'):
            try:
                batch = aggregate_batch.saved(path.parent.name)
                for index, row in enumerate(batch['results']):
                    record_id = f'{path.parent.name}-{index}'
                    if row['status'] != 'completed' or record_id in deleted:
                        continue
                    result = row['result']
                    records.append({'id': record_id, 'source': row['source'],
                                    'created_at': result.get('generated_at', ''),
                                    'count': result.get('count', 0),
                                    'thumbnail': result.get('result_url', '')})
            except (OSError, ValueError, KeyError):
                continue
        return sorted(records, key=lambda row: row['created_at'], reverse=True)


def delete(record_ids: list[str]) -> dict:
    # Hide individual records without rewriting the worker's live batch manifest.
    # Batch snapshots remain available to the original batch for recovery.
    with _lock:
        deleted = _deleted()
        missing, removed = [], []
        for record_id in dict.fromkeys(record_ids):
            try:
                get(record_id)
            except FileNotFoundError:
                missing.append(record_id)
            else:
                removed.append(record_id)
        deleted.update(removed)
        aggregate.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path = aggregate.RESULT_DIR / 'history-deleted.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(sorted(deleted)), encoding='utf-8')
        temp.replace(path)
        return {'deleted': removed, 'missing': missing}


def _image_data(batch_id: str, kind: str, filename: str) -> str:
    if not filename or '/' in filename or '\\' in filename or filename in ('.', '..'):
        raise ValueError('图片路径无效')
    path = aggregate_batch.directory(batch_id) / kind / filename
    mime = mimetypes.guess_type(filename)[0] or 'image/png'
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'data:{mime};base64,{encoded}'


def export(record_ids: list[str]) -> bytes:
    records = []
    for record_id in dict.fromkeys(record_ids):
        row = get(record_id)
        batch_id = record_id.split('-')[0]
        result = row['result']
        result['source_url'] = _image_data(batch_id, 'sources', row['source'])
        result['result_url'] = _image_data(batch_id, 'results', result['result'])
        records.append(row)
    payload = {'schema': 'aggregate-results', 'version': 1, 'records': records}
    viewer = Path(__file__).parent / 'frontend' / 'export' / 'aggregate-viewer.html'
    json_data = json.dumps(payload, ensure_ascii=False, indent=2)
    embedded = json_data.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    viewer_html = viewer.read_text(encoding='utf-8').replace(
        '<script id="embedded-results" type="application/json">null</script>',
        f'<script id="embedded-results" type="application/json">{embedded}</script>', 1)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('results.json', json_data)
        bundle.writestr('viewer.html', viewer_html)
    return archive.getvalue()
