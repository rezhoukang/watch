"""Logical upload paths keep existing flat image APIs compatible."""
import json
import re
import threading
import uuid
from pathlib import PurePosixPath

from fastapi import HTTPException, UploadFile
import cv2
import numpy as np
from . import common

_lock = threading.RLock()


def _read() -> dict:
    path = common.UPLOAD_DIR.parent / 'image-folders.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if 'paths' not in data:
        data = {'paths': data, 'folders': []}
    directories = set(data.get('folders', []))
    for relative in data['paths'].values():
        parts = relative.split('/')[:-1]
        directories.update('/'.join(parts[:index]) for index in range(1, len(parts) + 1))
    data['folders'] = sorted(directories)
    return data


def _save(data: dict) -> None:
    path = common.UPLOAD_DIR.parent / 'image-folders.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def paths() -> dict[str, str]:
    with _lock:
        return _read()['paths']


def folders() -> list[str]:
    with _lock:
        return _read()['folders']


def create_folder(name: str, parent: str = '') -> dict:
    name = name.strip()
    if not name or len(name) > 80 or name in ('.', '..') or re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or name.endswith('.'):
        raise HTTPException(status_code=400, detail='文件夹名称需为 1～80 个字符，不能包含文件名特殊字符或以点结尾')
    with _lock:
        data = _read()
        if parent and parent not in data['folders']:
            raise HTTPException(status_code=404, detail='父文件夹不存在')
        folder = f'{parent}/{name}' if parent else name
        if folder.casefold() in {item.casefold() for item in data['folders']}:
            raise HTTPException(status_code=400, detail='已存在同名文件夹')
        data['folders'].append(folder)
        _save(data)
        return {'path': folder, 'name': name}


def move(filenames: list[str], target: str = '') -> dict:
    with _lock:
        data = _read()
        if target and target not in data['folders']:
            raise HTTPException(status_code=404, detail='目标文件夹不存在')
        # Validate all sources before updating the mapping.
        for filename in filenames:
            common._safe_file(common.UPLOAD_DIR, filename)
        for filename in filenames:
            name = PurePosixPath(data['paths'].get(filename, filename)).name
            data['paths'][filename] = f'{target}/{name}' if target else name
        _save(data)
        return {'ok': True}


def upload(file: UploadFile, relative_path: str) -> dict:
    parts = relative_path.split('/')
    if len(parts) < 2 or len(relative_path) > 2000 or any(
            not part or part in ('.', '..') or re.search(r'[\\:\x00-\x1f]', part) for part in parts):
        raise HTTPException(status_code=400, detail='非法文件夹路径')
    suffix = PurePosixPath(relative_path).suffix.lower()
    if suffix not in common.IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail='文件夹上传仅接受图片')
    try:
        image = cv2.imdecode(np.frombuffer(file.file.read(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError('无法解码图片')
        file.file.seek(0)
    except (OSError, ValueError, cv2.error) as error:
        raise HTTPException(status_code=400, detail='不是有效的图片文件') from error
    file.filename = f'{uuid.uuid4().hex}{suffix}'
    filename, _ = common._save_upload(file, common.UPLOAD_DIR, common.IMAGE_EXTENSIONS)
    try:
        with _lock:
            data = _read()
            data['paths'][filename] = relative_path
            _save(data)
    except Exception:
        (common.UPLOAD_DIR / filename).unlink(missing_ok=True)
        raise
    return {'filename': filename, 'relative_path': relative_path, 'url': f'/images/{filename}'}
