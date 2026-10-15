"""Face library management and 1:N recognition API routes."""
import cv2
import numpy as np

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

import face_recognition
import head_cut_faces
from project_paths import IMAGE_EXTENSIONS

router = APIRouter()


class RecognizeRequest(BaseModel):
    threshold: float = Field(default=face_recognition.DEFAULT_MATCH_THRESHOLD, ge=0.0, le=1.0)
    min_confidence: float = Field(default=face_recognition.DEFAULT_QUALITY_CONFIDENCE, ge=0.0, le=1.0)
    min_size: int = Field(default=face_recognition.DEFAULT_MIN_FACE_PIXELS, ge=1, le=1000)
    min_sharpness: float = Field(default=face_recognition.DEFAULT_MIN_SHARPNESS, ge=0.0, le=10000.0)


@router.get("/api/face-library/status", summary="识别模型与人脸库状态")
def status() -> dict:
    return face_recognition.status()


@router.get("/api/face-library", summary="人脸库列表")
def list_library() -> dict:
    return {"entries": face_recognition.list_entries()}


@router.post("/api/face-library", summary="添加一个人到人脸库")
def add_library_entry(
    name: str = Form(..., min_length=1, max_length=80),
    image: UploadFile = File(...),
) -> dict:
    suffix = (image.filename or "").lower()
    ext = "." + suffix.rsplit(".", 1)[-1] if "." in suffix else ""
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"仅支持图片：{', '.join(sorted(IMAGE_EXTENSIONS))}")
    data = image.file.read()
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="无法解析上传的图片")
    try:
        return face_recognition.add_entry(name, decoded, source="upload")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/api/face-library/{entry_id}", summary="从人脸库删除一个人")
def delete_library_entry(entry_id: str) -> dict:
    if not face_recognition.delete_entry(entry_id):
        raise HTTPException(status_code=404, detail="人脸库中不存在该编号")
    return {"ok": True}


@router.post("/api/face-library/import-students", summary="导入2026软件学生花名册")
def import_students() -> dict:
    try:
        return face_recognition.import_students()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/head-cuts/{run_id}/recognize", summary="对已检出的全部人脸做1:N识别")
def recognize_run_faces(run_id: str, payload: RecognizeRequest) -> dict:
    """对批次中每个 crop 已检出的人脸裁剪做识别，结果写回 recognition.json。"""
    try:
        job_id = head_cut_faces.start_recognition(run_id, payload.threshold, payload.min_confidence,
                                                  payload.min_size, payload.min_sharpness)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job_id, "run_id": run_id}
