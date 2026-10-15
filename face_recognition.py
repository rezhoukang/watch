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
# 库内登记照：对齐后的人脸图，供识别结果里人工核对“这张脸对应的库里是谁”
PHOTO_DIR = LIBRARY_DIR / "photos"
INPUT_SIZE = 112

# 学生花名册快照（编号→姓名）。仅在导入时使用，运行中以后台任务同步。
STUDENTS_PDF_DIR = BASE_DIR / "output" / "pdf" / "2026_software_students"

# 余弦相似度匹配：低于该阈值视为“未登记/不识别”（点名宁可漏不可错）。
DEFAULT_MATCH_THRESHOLD = 0.40

# 识别查询端的质量门槛默认值（均可调，不再是硬编码）。
# min_confidence 取 SCRFD 检出阈值 0.5：检测已按该阈值筛过，再要求更高置信度等于重复设卡；
# 实测 A/B 教室远景小脸的 SCRFD 置信度中位 0.58、最高仅 0.774，卡 0.8 会让全部人脸进不了比对。
DEFAULT_QUALITY_CONFIDENCE = 0.5
DEFAULT_MIN_FACE_PIXELS = 24
DEFAULT_MIN_SHARPNESS = 50
# 识别端 SCRFD 检出阈值（与 DEFAULT_QUALITY_CONFIDENCE 保持同源）
DETECT_CONFIDENCE = 0.5

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
            "quality": {"min_confidence": DEFAULT_QUALITY_CONFIDENCE,
                        "min_size": DEFAULT_MIN_FACE_PIXELS,
                        "min_sharpness": DEFAULT_MIN_SHARPNESS},
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
    outputs = session.run([output_name], {"input.1": blob})
    embedding = np.asarray(outputs[0], dtype=np.float32)[0]
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
    save_photo(entry_id, image, face["landmarks"])
    return {"id": entry_id, "name": name}


def _next_id(entries: dict[str, dict]) -> str:
    used = {int(k) for k in entries if k.isdigit()}
    index = 1
    while index in used:
        index += 1
    return str(index).zfill(4)


def photo_path(entry_id: str) -> Path | None:
    """库内登记照路径。entry_id 只允许数字，防目录穿越。"""
    if not isinstance(entry_id, str) or not re.fullmatch(r"\d{1,8}", entry_id):
        return None
    return PHOTO_DIR / f"{entry_id}.jpg"


def save_photo(entry_id: str, image: np.ndarray, landmarks) -> bool:
    """按五点对齐后存一张登记照（112px 输入尺度）。对齐失败则跳过，不影响入库。"""
    target = photo_path(entry_id)
    if target is None:
        return False
    try:
        aligned = align_face(_to_bgr(image), landmarks)
    except (ValueError, TypeError, cv2.error):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", aligned, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return False
    encoded.tofile(str(target))
    return True


def photo_file(entry_id: str) -> Path | None:
    """返回已存在的库内登记照；不存在或编号非法则 None。"""
    with _lock:
        if entry_id not in _load_entries():
            return None
    path = photo_path(entry_id)
    return path if path is not None and path.is_file() else None


def photo_url(entry_id: str) -> str | None:
    """库内登记照的接口地址（无照片时 None）。"""
    path = photo_path(entry_id)
    return f"/api/face-library/{entry_id}/photo" if path is not None and path.is_file() else None


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
    path = photo_path(entry_id)
    if path is not None and path.is_file():
        path.unlink(missing_ok=True)
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
        "photo_url": photo_url(ids[index]),
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
    return [{"entry_id": i, "name": entries.get(i, {}).get("name"), "photo_url": photo_url(i),
             "similarity": round(s, 4)} for i, s in ranked[:k]]


# 冲突消解时每个脸取前 K 个候选参与分配（多留几个，便于被抢后往下顺延）
CANDIDATE_K = 5


# ------------------------------------------------------------------ 一对一冲突消解

def resolve_one_to_one(candidates_by_face: dict[str, list[dict]], threshold: float) -> dict[str, dict]:
    """把“多张脸抢同一个库内编号”按相似度降序贪心消解成一对一分配。

    规则：
    - 只有相似度 ≥ threshold 的候选才有资格参与分配；
    - 所有 (脸，候选) 组合按相似度降序处理，**相似度高的先占**；
    - 一个库内编号只能被一张脸占用；脸被抢后自动顺延到它的下一个候选，
      再被抢就继续顺延（递归），直到没有可用候选 → 该脸判为未识别；
    - 相似度完全并列时按传入顺序先到先得（list.sort 稳定），保证结果可复现。

    返回 {face_id: {'assigned': 候选(含 rank) | None, 'skipped': [被占用的候选…]}}，
    其中 skipped 里的 taken_by 记下是谁抢走的，便于界面解释。
    """
    pairs: list[tuple[float, str, int, dict]] = []
    for face_id, candidates in candidates_by_face.items():
        for rank, candidate in enumerate(candidates, 1):
            if candidate.get("similarity", 0.0) >= threshold:
                pairs.append((candidate["similarity"], face_id, rank, candidate))
    pairs.sort(key=lambda item: item[0], reverse=True)

    taken: dict[str, str] = {}
    outcome = {face_id: {"assigned": None, "skipped": []} for face_id in candidates_by_face}
    for similarity, face_id, rank, candidate in pairs:
        if outcome[face_id]["assigned"] is not None:
            continue  # 已经拿到了得分更高的候选
        entry_id = candidate["entry_id"]
        holder = taken.get(entry_id)
        if holder is not None:
            outcome[face_id]["skipped"].append({
                "entry_id": entry_id, "name": candidate.get("name"),
                "similarity": similarity, "rank": rank, "taken_by": holder,
            })
            continue
        taken[entry_id] = face_id
        outcome[face_id]["assigned"] = {**candidate, "rank": rank}
    return outcome


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

    已存在且已有登记照的编号跳过（幂等）；已入库但缺登记照的老条目会自动回填照片。
    progress 是可选回调 (done, total, message)。
    兼容两种文件名：`001_示例姓名.jpeg`（带姓名）或 `001.jpeg`（纯编号）。
    姓名优先取文件名，回退到 names.json。
    返回 {imported, skipped, photos_added, existing, errors, total}。
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
    imported, skipped, photos_added, errors = 0, 0, 0, []
    total = len(files)
    done = 0
    detector = _get_detector()
    for index in sorted(files, key=int):
        done += 1
        number = int(index)
        entry_id = str(index).zfill(4)
        path, name_from_file = files[index]
        name = (name_from_file or names.get(index, "")).strip()
        already = number in existing
        photo = photo_path(entry_id)
        has_photo = photo is not None and photo.is_file()
        # 已入库且已有登记照才算真正跳过；缺照片的老条目要回填
        if already and has_photo:
            skipped += 1
            if progress:
                progress(done, total, f"跳过已存在 {index} {names.get(index, '')}")
            continue
        try:
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("无法读取图片")
            faces = detector.detect(image, threshold=0.5)
            if not faces:
                raise ValueError("未检测到人脸")
            faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
            face = faces[0]
            if not already:
                embedding = extract_embedding(image, face["landmarks"])
                with _lock:
                    entries = _load_entries()
                    entries[entry_id] = {
                        "id": entry_id, "name": name or f"学生{index}",
                        "source": "students-2026", "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    ids, matrix = _load_embeddings()
                    ids = ids + [entry_id]
                    matrix = np.vstack([matrix, embedding[None, :]]) if matrix.size else embedding[None, :]
                    _write_entries(entries)
                    _write_embeddings(ids, matrix)
                imported += 1
            if not has_photo:
                if save_photo(entry_id, image, face["landmarks"]):
                    photos_added += 1
            if progress:
                progress(done, total, f"{'已导入' if not already else '已补登记照'} {index} {name}")
        except Exception as exc:  # noqa: BLE001
            errors.append({"index": index, "name": name, "message": str(exc)})
            if progress:
                progress(done, total, f"导入失败 {index} {name}：{exc}")
    return {"imported": imported, "skipped": skipped, "photos_added": photos_added,
            "existing": existing, "errors": errors, "total": total}


# ---------------------------------------------------------------- 识别一个人脸裁剪

def recognize_face_image(image: np.ndarray, threshold: float = DEFAULT_MATCH_THRESHOLD,
                         min_confidence: float = DEFAULT_QUALITY_CONFIDENCE,
                         min_size: int = DEFAULT_MIN_FACE_PIXELS,
                         min_sharpness: float = DEFAULT_MIN_SHARPNESS) -> dict:
    """对一张人脸图做 1:N 识别。image 为 BGR；内部自动检测+对齐+提特征+比对。

    质量门槛全部可调：min_confidence / min_size / min_sharpness。
    """
    image = _to_bgr(image)
    detector = _get_detector()
    faces = detector.detect(image, threshold=DETECT_CONFIDENCE)
    if not faces:
        return {"status": "no_face", "face_count": 0, "matches": []}
    faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
    results = []
    for face in faces[:5]:
        reasons, _quality = assess_quality(image, face, min_size=min_size,
                                           min_sharpness=min_sharpness,
                                           min_confidence=min_confidence)
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
