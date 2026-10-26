"""Independent action recognition: Head expansion, editable crops and local VLM jobs."""
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from action_recognition import jobs, store, vlm
from action_recognition.regions import Expansion
from project_paths import IMAGE_EXTENSIONS
from . import common

router = APIRouter(prefix="/api/actions", tags=["独立动作识别"])


class ExpansionRequest(BaseModel):
    side: float = Field(default=1.0, ge=0, le=5, allow_inf_nan=False)
    top: float = Field(default=0.0, ge=0, le=3, allow_inf_nan=False)
    below: float = Field(default=1.3, ge=0, le=12, allow_inf_nan=False)

    def expansion(self) -> Expansion:
        return Expansion(self.side, self.top, self.below)


class PrepareRequest(ExpansionRequest):
    filenames: list[str] = Field(min_length=1, max_length=100)


class RegionRequest(BaseModel):
    bbox: list[float] = Field(min_length=4, max_length=4)


class ModelRequest(BaseModel):
    model: Literal["4b", "2b"] = "4b"
    quantization: Literal["4bit", "fp16"] = "4bit"


class AnalyzeRequest(ModelRequest):
    crop_ids: list[str] = Field(default_factory=list, max_length=2000)
    force: bool = False


def _call(function, *args):
    try:
        return function(*args)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/status")
def status(model: Literal["4b", "2b"] = "4b", quantization: Literal["4bit", "fp16"] = "4bit"):
    return {**vlm.status(model, quantization), "active_job": jobs.active()}


@router.post("/model/download")
def download_model(payload: ModelRequest):
    return _call(jobs.start_download, payload.model)


@router.post("/model/unload")
def unload_model():
    try:
        with jobs.mutation():
            vlm.unload()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"ok": True}


@router.get("/runs")
def list_runs():
    return {"runs": store.list_runs()}


@router.post("/runs")
def prepare(payload: PrepareRequest):
    paths = [common._safe_file(common.UPLOAD_DIR, name) for name in dict.fromkeys(payload.filenames)]
    if any(path.suffix.lower() not in IMAGE_EXTENSIONS for path in paths):
        raise HTTPException(status_code=400, detail="只能选择待推理列表中的图片")
    common._require_head_detection()
    return _call(jobs.start_prepare, paths, payload.expansion())


@router.get("/runs/{run_id}")
def read_run(run_id: str):
    return _call(store.read, run_id)


@router.post("/runs/{run_id}/expansion")
def update_expansion(run_id: str, payload: ExpansionRequest):
    try:
        with jobs.mutation():
            return _call(store.reexpand, run_id, payload.expansion())
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.patch("/runs/{run_id}/crops/{crop_id}")
def update_crop(run_id: str, crop_id: str, payload: RegionRequest):
    try:
        with jobs.mutation():
            return _call(store.update_region, run_id, crop_id, payload.bbox)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/runs/{run_id}/analyze")
def analyze(run_id: str, payload: AnalyzeRequest):
    return _call(jobs.start_analysis, run_id, payload.crop_ids, payload.model, payload.quantization, payload.force)


@router.get("/runs/{run_id}/files/{filename:path}")
def image_file(run_id: str, filename: str):
    path = _call(store.image_file, run_id, filename)
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    return _call(jobs.get, job_id)


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str):
    return _call(jobs.cancel, job_id)
