"""1:N face recognition: feature extraction, dynamic library and matching.

Detection/landmarks reuse ScrfdDetector; alignment reuses align_face. The
recognition model is insightface buffalo_l's w600k_r50.onnx (512-D embedding),
which is exactly the companion of the already-used det_10g.onnx.
"""
import json
import math
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from project_paths import BASE_DIR, IMAGE_EXTENSIONS
from scrfd_detector import ScrfdDetector, align_face, assess_quality, MODEL_PATH, MODEL_SHA256

MODEL_PATH_RECOGNITION = Path(BASE_DIR) / "weight" / "scrfd" / "w600k_r50.onnx"
LIBRARY_DIR = BASE_DIR / "output" / "face_library"
LIBRARY_JSON = LIBRARY_DIR / "library.json"
LIBRARY_NPZ = LIBRARY_DIR / "embeddings.npz"
INPUT_SIZE = 112

# 学生花名册快照（编号→姓名）。仅在导入时使用，运行中以后台任务同步。
STUDENTS_PDF_DIR = BASE_DIR / "output" / "pdf" / "2026_software_students"

# 余弦相似度匹配：低于该阈值视为“未登记/不识别”（点名宁可漏不可错）。
DEFAULT_MATCH_THRESHOLD = 0.40

_lock = threading.RLock()
_session = None
_detector = None


def _get_session():
    """Lazily load the recognition ONNX session (CPU)."""
    global _session
    if _session is None:
        import onnxruntime as ort

        if not MODEL_PATH_RECOGNITION.is_file():
            raise FileNotFoundError(f"识别模型不存在：{MODEL_PATH_RECOGNITION}（请放入 weight/scrdf/w600k_r50.onnx）")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        _session = ort.InferenceSession(str(MODEL_PATH_RECOGNITION), sess_options=options,
                                        providers=["CPUExecutionProvider"])
    return _session


def _get_detector() -> ScrfdDetector:
    global _detector
    if _detector is None:
        _detector = ScrfdDetector()
    return _detector


def status() -> dict:
    """识别模型与当前人脸库的可用状态（供界面提示）。"""
    with _lock:
        return {
            "available": MODEL_PATH_RECOGNITION.is_file(),
            "model": "w600k_r50.onnx",
            "model_sha256": None,
            "detector": {"available": MODEL_PATH.is_file(), "name": "SCRFD-10G", "sha256": MODEL_SHA256},
            "library_count": len(_load_entries()),
            "threshold": DEFAULT_MATCH_THRESHOLD,
        }


# ---------------------------------------------------------------- 特征提取

def extract_embedding(face_image, landmarks) -> np.ndarray:
    """从一张人脸图（BGR）+ 五点关键点提取 512 维单位向量。

    - 先按 SCRFD 五点用 align_face 对齐（224×224），再缩放到 112×112。
    - 与 insightface buffalo_l 的 ArcFace 前处理一致：归一化到 [-1,1]。
    """
    aligned = align_face(face_image, landmarks)
    resized = cv2.resize(aligned, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    blob = np.ascontiguousarray(resized).astype(np.float32) / 127.5 - 1.0
    blob = blob.transpose(2, 0, 1)[None, ...]
    session = _get_session()
    output_name = session.get_outputs()[0].name
    embedding = session.run([output_name], {"input.1": blob})[0][0]
    norm = float(np.linalg.norm(embedding))
    if norm < 1e-9:
        raise ValueError("特征向量模长异常")
    return embedding / norm


# ---------------------------------------------------------------- 人脸库

def _load_entries() -> dict[str, dict]:
    """{entry_id: {id, name, created_at, source}}。文件缺失时返回空库。"""
    if not LIBRARY_JSON.is_file():
        return {}
    return json.loads(LIBRARY_JSON.read_text(encoding="utf-8"))


def _write_entries(entries: dict[str, dict]) -> None:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    temporary = LIBRARY_JSON.with_suffix(LIBRARY_JSON.suffix + ".tmp")
    temporary.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(LIBRARY_JSON)


def _write_embeddings(ids: list[str], matrix: np.ndarray) -> None:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    temporary = LIBRARY_NPZ.with_suffix(LIBRARY_NPZ.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, ids=np.asarray(ids, dtype=object), matrix=matrix.astype(np.float32))
    temporary.replace(LIBRARY_NPZ)


def _load_embeddings() -> tuple[list[str], np.ndarray]:
    """返回 (ids, matrix[N,512])。无库时 ids=[]、matrix 为空 (0,512)。"""
    if not LIBRARY_NPZ.is_file():
        return [], np.zeros((0, 512), dtype=np.float32)
    data = np.load(LIBRARY_NPZ, allow_pickle=True)
    return [str(v) for v in data["ids"]], np.asarray(data["matrix"], dtype=np.float32)


def list_entries() -> list[dict]:
    with _lock:
        entries = _load_entries()
        return [dict(value, id=key) for key, value in entries.items()]


def add_entry(name: str, image: np.ndarray, source: str = "upload") -> dict:
    """向库中添加一个人（一张登记照），自动检测、对齐并提特征。

    建库照片是权威基准，只要求人脸检测成功（置信度≥0.5）即可入库；
    不做清晰度/几何硬筛选（那是查询端的防误报关卡）。
    """
    name = (name or "").strip()
    if not name or len(name) > 80:
        raise ValueError("姓名不能为空且不超过80字")
    image = _to_bgr(image)
    detector = _get_detector()
    faces = detector.detect(image, threshold=0.5)
    if not faces:
        raise ValueError("照片中未检测到人脸，请上传正面清晰照片")
    faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
    face = faces[0]
    embedding = extract_embedding(image, face["landmarks"])
    with _lock:
        entries = _load_entries()
        entry_id = _next_id(entries)
        entries[entry_id] = {
            "id": entry_id, "name": name, "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        ids, matrix = _load_embeddings()
        ids = ids + [entry_id]
        matrix = np.vstack([matrix, embedding[None, :]]) if matrix.size else embedding[None, :]
        _write_entries(entries)
        _write_embeddings(ids, matrix)
    return {"id": entry_id, "name": name}


def _next_id(entries: dict[str, dict]) -> str:
    used = {int(k) for k in entries if k.isdigit()}
    index = 1
    while index in used:
        index += 1
    return str(index).zfill(4)


def delete_entry(entry_id: str) -> bool:
    with _lock:
        entries = _load_entries()
        if entry_id not in entries:
            return False
        del entries[entry_id]
        ids, matrix = _load_embeddings()
        keep = [i for i, v in enumerate(ids) if v != entry_id]
        _write_entries(entries)
        _write_embeddings([ids[i] for i in keep], matrix[keep] if keep else np.zeros((0, 512), dtype=np.float32))
        return True


def _to_bgr(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if image.shape[2] == 3:
            return image
    raise ValueError("不支持的图片格式")


# ---------------------------------------------------------------- 1:N 比对

def match(embedding: np.ndarray, threshold: float = DEFAULT_MATCH_THRESHOLD) -> dict:
    """在库中找最优匹配。返回 {matched, entry_id, name, similarity}。

    阈值以下视为“未登记”（matched=False，但仍返回最高分供参考）。
    """
    with _lock:
        entries = _load_entries()
        ids, matrix = _load_embeddings()
    if not ids:
        return {"matched": False, "entry_id": None, "name": None, "similarity": 0.0}
    scores = matrix @ embedding.reshape(-1)
    index = int(np.argmax(scores))
    best = float(scores[index])
    entry = entries.get(ids[index], {})
    return {
        "matched": best >= threshold,
        "entry_id": ids[index],
        "name": entry.get("name"),
        "similarity": round(best, 4),
    }


def match_top_k(embedding: np.ndarray, k: int = 3) -> list[dict]:
    """返回库中相似度最高的前 k 名（不应用阈值，用于界面候选展示）。"""
    with _lock:
        entries = _load_entries()
        ids, matrix = _load_embeddings()
    if not ids:
        return []
    scores = (matrix @ embedding.reshape(-1)).tolist()
    ranked = sorted(zip(ids, scores), key=lambda item: item[1], reverse=True)
    return [{"entry_id": i, "name": entries.get(i, {}).get("name"),
             "similarity": round(s, 4)} for i, s in ranked[:k]]


# ---------------------------------------------------------------- 学生花名册批量导入

def _load_student_names() -> dict[str, str]:
    """读取 output/pdf/2026_software_students/names.json，返回 {编号(3位): 姓名}。"""
    path = STUDENTS_PDF_DIR / "names.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for student in data.get("students", []):
        index = int(student.get("index", 0))
        if 1 <= index <= 999:
            result[str(index).zfill(3)] = student.get("name", "").strip()
    return result


def import_students(progress=None) -> dict:
    """把 2026_software_students/photos/*.jpeg 全部导入人脸库（编号→姓名）。

    已存在的编号跳过（幂等）。progress 是可选回调 (done, total, message)。
    兼容两种文件名：`001_示例姓名.jpeg`（带姓名）或 `001.jpeg`（纯编号）。
    姓名优先取文件名，回退到 names.json。返回 {imported, skipped, existing, errors}。
    """
    names = _load_student_names()
    photos = STUDENTS_PDF_DIR / "photos"
    if not photos.is_dir():
        raise FileNotFoundError(f"学生照片目录不存在：{photos}")
    files = {}
    for path in photos.iterdir():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = re.fullmatch(r"(\d{3})(?:_([^.]*))?\.(jpe?g|png|bmp|webp)", path.name, re.I)
        if match:
            files[match.group(1)] = (path, match.group(2) or "")
    with _lock:
        entries = _load_entries()
    existing = {int(k) for k in entries if k.isdigit() and int(k) <= 999}
    imported, skipped, errors = 0, 0, []
    total = len(files)
    done = 0
    detector = _get_detector()
    for index in sorted(files, key=int):
        done += 1
        number = int(index)
        if number in existing:
            skipped += 1
            if progress:
                progress(done, total, f"跳过已存在 {index} {names.get(index, '')}")
            continue
        path, name_from_file = files[index]
        name = (name_from_file or names.get(index, "")).strip()
        try:
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("无法读取图片")
            faces = detector.detect(image, threshold=0.5)
            if not faces:
                raise ValueError("未检测到人脸")
            faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
            face = faces[0]
            embedding = extract_embedding(image, face["landmarks"])
            with _lock:
                entries = _load_entries()
                entries[str(index).zfill(4)] = {
                    "id": str(index).zfill(4), "name": name or f"学生{index}",
                    "source": "students-2026", "created_at": datetime.now(timezone.utc).isoformat(),
                }
                ids, matrix = _load_embeddings()
                entry_id = str(index).zfill(4)
                ids = ids + [entry_id]
                matrix = np.vstack([matrix, embedding[None, :]]) if matrix.size else embedding[None, :]
                _write_entries(entries)
                _write_embeddings(ids, matrix)
            imported += 1
            if progress:
                progress(done, total, f"已导入 {index} {name}")
        except Exception as exc:  # noqa: BLE001
            errors.append({"index": index, "name": name, "message": str(exc)})
            if progress:
                progress(done, total, f"导入失败 {index} {name}：{exc}")
    return {"imported": imported, "skipped": skipped, "existing": existing, "errors": errors, "total": total}


# ---------------------------------------------------------------- 识别一个人脸裁剪

def recognize_face_image(image: np.ndarray, threshold: float = DEFAULT_MATCH_THRESHOLD) -> dict:
    """对一张人脸图做 1:N 识别。image 为 BGR；内部自动检测+对齐+提特征+比对。"""
    image = _to_bgr(image)
    detector = _get_detector()
    faces = detector.detect(image, threshold=0.5)
    if not faces:
        return {"status": "no_face", "face_count": 0, "matches": []}
    faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
    results = []
    for face in faces[:5]:
        reasons, _quality = assess_quality(image, face)
        if reasons:
            results.append({"status": "rejected", "reasons": reasons, "matches": []})
            continue
        try:
            embedding = extract_embedding(image, face["landmarks"])
            match_result = match(embedding, threshold)
            results.append({
                "status": "matched" if match_result["matched"] else "unknown",
                "match": match_result,
                "candidates": match_top_k(embedding, k=3),
            })
        except Exception as exc:  # noqa: BLE001
            results.append({"status": "error", "message": str(exc), "matches": []})
    return {"status": "ok", "face_count": len(results), "faces": results}
