"""Saved per-image aggregate results, backed by completed batch snapshots."""
import base64
import copy
import io
import json
import mimetypes
import re
import threading
import zipfile
import uuid
from pathlib import Path
from urllib.parse import quote
from PIL import Image, ImageOps

import aggregate
import aggregate_batch

_lock = threading.RLock()
COMPRESSION = {
    'balanced': {'sources': (1600, 80), 'results': (640, 75)},
    'compact': {'sources': (1280, 65), 'results': (480, 60)},
}


def _folder_state() -> dict:
    path = aggregate.RESULT_DIR / 'history-folders.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    return {'folders': data, 'classified_records': []} if isinstance(data, list) else data


def folders() -> list[dict]:
    with _lock:
        return _folder_state()['folders']


def _save_folders(items: list[dict], classified_records: list[str] | None = None) -> None:
    aggregate.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = aggregate.RESULT_DIR / 'history-folders.json'
    temp = path.with_suffix('.tmp')
    state = _folder_state()
    state['folders'] = items
    if classified_records is not None:
        state['classified_records'] = classified_records
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def descendant_ids(items: list[dict], roots: list[str]) -> set[str]:
    found = set(roots)
    pending = list(roots)
    while pending:
        parent = pending.pop()
        for item in items:
            if item.get('parent_id') == parent and item['id'] not in found:
                found.add(item['id'])
                pending.append(item['id'])
    return found


def create_folder(name: str, parent_id: str | None = None) -> dict:
    name = name.strip()
    reserved = re.fullmatch(r'(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?', name, re.IGNORECASE)
    if not name or len(name) > 80 or re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or name.endswith('.') or reserved:
        raise ValueError('文件夹名称需为 1～80 个字符，且不能包含文件名特殊字符或以点结尾')
    with _lock:
        items = folders()
        if parent_id and not any(item['id'] == parent_id for item in items):
            raise FileNotFoundError('父文件夹不存在')
        if any(item['name'].casefold() == name.casefold() and item.get('parent_id') == parent_id for item in items):
            raise ValueError('已存在同名文件夹')
        folder = {'id': uuid.uuid4().hex, 'name': name, 'parent_id': parent_id, 'record_ids': []}
        _save_folders([*items, folder])
        return folder


def move_to_folder(record_ids: list[str], folder_id: str | None, folder_ids: list[str] | None = None) -> None:
    with _lock:
        items = folders()
        target = next((item for item in items if item['id'] == folder_id), None)
        if folder_id and target is None:
            raise FileNotFoundError('文件夹不存在')
        selected_folders = set(folder_ids or [])
        if not selected_folders.issubset({item['id'] for item in items}):
            raise FileNotFoundError('待移动的文件夹不存在')
        moving_tree = descendant_ids(items, list(selected_folders))
        if folder_id in moving_tree:
            raise ValueError('不能把文件夹移入自身或其子文件夹')
        ids = list(dict.fromkeys(record_ids))
        for record_id in ids:
            get(record_id)
        moving = set(ids)
        for item in items:
            item['record_ids'] = [record_id for record_id in item['record_ids'] if record_id not in moving]
        if target is not None:
            target['record_ids'].extend(ids)
        for item in items:
            if item['id'] in selected_folders and item.get('parent_id') not in moving_tree:
                item['parent_id'] = folder_id
        classified = set(_folder_state().get('classified_records', [])) | moving
        _save_folders(items, sorted(classified))


def sync_source_folders(source_paths: dict[str, str]) -> list[dict]:
    """Mirror source directories once, preserving subsequent manual classification."""
    with _lock:
        records = listing()
        state = _folder_state()
        processed = set(state.get('classified_records', []))
        pending = [row for row in records if row['id'] not in processed]
        if not pending:
            return records
        items = state['folders']
        assigned = {record_id for item in items for record_id in item['record_ids']}
        lookup = {(item.get('parent_id'), item['name'].casefold()): item for item in items}
        for row in pending:
            processed.add(row['id'])
            if row['id'] in assigned:
                continue
            relative = row.get('relative_path') or source_paths.get(
                row['source_file'], source_paths.get(row['source'], row['source']))
            parts = relative.split('/')[:-1]
            if not parts or any(part in ('', '.', '..') for part in parts):
                continue
            parent = None
            for name in parts:
                key = (parent, name.casefold())
                if key not in lookup:
                    folder = {'id': uuid.uuid4().hex, 'name': name, 'parent_id': parent, 'record_ids': []}
                    items.append(folder)
                    lookup[key] = folder
                parent = lookup[key]['id']
            lookup[key]['record_ids'].append(row['id'])
        _save_folders(items, sorted(processed))
        return records


def export_selection(record_ids: list[str], folder_ids: list[str], compression: str = 'original') -> tuple[bytes, str]:
    with _lock:
        items = {item['id']: item for item in folders()}
        ids = list(record_ids)
        deleted = _deleted()
        for folder_id in dict.fromkeys(folder_ids):
            if folder_id not in items:
                raise FileNotFoundError('文件夹不存在')
            for descendant in descendant_ids(list(items.values()), [folder_id]):
                ids.extend(record_id for record_id in items[descendant]['record_ids'] if record_id not in deleted)
        if not ids:
            raise ValueError('所选文件夹中没有可导出的记录')
        name = items[folder_ids[0]]['name'] if len(folder_ids) == 1 and not record_ids else None
        return export(ids, name, compression), (name or 'aggregate-results') + '.zip'


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
            'source_file': row.get('source_file', row['source']),
            'relative_path': row.get('relative_path'),
            'count': result.get('count', 0), 'result': result}


def get(record_id: str) -> dict:
    with _lock:
        if record_id in _deleted():
            raise FileNotFoundError('聚合记录已删除')
        return _read(record_id)


def selected_records(record_ids: list[str], folder_ids: list[str]) -> list[dict]:
    with _lock:
        items = folders()
        if not set(folder_ids).issubset({item['id'] for item in items}):
            raise FileNotFoundError('文件夹不存在')
        branch = descendant_ids(items, folder_ids)
        ids = list(record_ids)
        deleted = _deleted()
        for folder in items:
            if folder['id'] in branch:
                ids.extend(record_id for record_id in folder['record_ids'] if record_id not in deleted)
        return [get(record_id) for record_id in dict.fromkeys(ids)]


def listing() -> list[dict]:
    with _lock:
        deleted = _deleted()
        records = []
        for path in (aggregate.RESULT_DIR / 'batches').glob('*/record.json'):
            try:
                batch = aggregate_batch.saved(path.parent.name)
                if batch.get('mirror'):
                    continue
                for index, row in enumerate(batch['results']):
                    record_id = f'{path.parent.name}-{index}'
                    if row['status'] != 'completed' or record_id in deleted:
                        continue
                    result = row['result']
                    records.append({'id': record_id, 'source': row['source'],
                                    'source_file': row.get('source_file', row['source']),
                                    'relative_path': row.get('relative_path'),
                                    'created_at': result.get('generated_at', ''),
                                    'count': result.get('count', 0),
                                    'thumbnail': result.get('result_url', '')})
            except (OSError, ValueError, KeyError):
                continue
        return sorted(records, key=lambda row: row['created_at'], reverse=True)


def delete(record_ids: list[str], folder_ids: list[str] | None = None) -> dict:
    # Hide individual records without rewriting the worker's live batch manifest.
    # Batch snapshots remain available to the original batch for recovery.
    with _lock:
        items = folders()
        existing_folders = {item['id'] for item in items}
        requested_folders = set(folder_ids or [])
        removing_folders = descendant_ids(items, list(requested_folders & existing_folders))
        ids = list(record_ids)
        for item in items:
            if item['id'] in removing_folders:
                ids.extend(item['record_ids'])
        deleted = _deleted()
        missing, removed = [], []
        for record_id in dict.fromkeys(ids):
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
        # Write tombstones first so an interrupted folder update cannot make
        # deleted records reappear as unclassified results. Retrying is safe.
        if removing_folders:
            _save_folders([item for item in items if item['id'] not in removing_folders])
        return {'deleted': removed, 'missing': missing,
                'deleted_folders': sorted(removing_folders),
                'missing_folders': sorted(requested_folders - existing_folders)}


def _image_data(batch_id: str, kind: str, filename: str, compression: str = 'original') -> str:
    if not filename or '/' in filename or '\\' in filename or filename in ('.', '..'):
        raise ValueError('图片路径无效')
    path = aggregate_batch.directory(batch_id) / kind / filename
    mime = mimetypes.guess_type(filename)[0] or 'image/png'
    data = path.read_bytes()
    if compression != 'original':
        edge, quality = COMPRESSION[compression][kind]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGBA')
            resized = max(image.size) > edge
            image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
            background = Image.new('RGB', image.size, 'white')
            background.paste(image, mask=image.getchannel('A'))
            output = io.BytesIO()
            background.save(output, format='JPEG', quality=quality, optimize=True)
        if resized or output.tell() < len(data):
            data, mime = output.getvalue(), 'image/jpeg'
    encoded = base64.b64encode(data).decode('ascii')
    return f'data:{mime};base64,{encoded}'


def export(record_ids: list[str], name: str | None = None, compression: str = 'original') -> bytes:
    if compression not in ('original', *COMPRESSION):
        raise ValueError('导出压缩模式无效')
    records = []
    for record_id in dict.fromkeys(record_ids):
        row = get(record_id)
        batch_id = record_id.split('-')[0]
        result = row['result']
        result['source_url'] = _image_data(batch_id, 'sources', row['source_file'], compression)
        result['result_url'] = _image_data(batch_id, 'results', result['result'], compression)
        records.append(row)
    payload = {'schema': 'aggregate-results', 'version': 1, 'records': records,
               'title': name or '聚合结果查看器', 'compression': compression}
    viewer = Path(__file__).parent / 'frontend' / 'export' / 'aggregate-viewer.html'
    json_data = json.dumps(payload, ensure_ascii=False, indent=2)
    embedded = json_data.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    viewer_html = viewer.read_text(encoding='utf-8').replace(
        '<script id="embedded-results" type="application/json">null</script>',
        f'<script id="embedded-results" type="application/json">{embedded}</script>', 1)
    json_filename = f'{name}.json' if name else 'results.json'
    html_filename = f'{name}.html' if name else 'viewer.html'
    viewer_html = viewer_html.replace('./results.json', './' + quote(json_filename, safe=''))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(json_filename, json_data)
        bundle.writestr(html_filename, viewer_html)
    return archive.getvalue()
