"""jobs: extracted API routes; URLs and response formats are unchanged."""
from fastapi import APIRouter, HTTPException
import vision
from . import common

router = APIRouter()


@router.post("/api/train", summary="启动训练（后台任务）")
def start_train(
    dataset: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    lr0: float = 0.01,
    weights: str = vision.DEFAULT_WEIGHTS,
) -> dict:
    common._require_yolo_weights(weights)
    available, message = vision.yolo_available()
    if not available:
        raise HTTPException(status_code=503, detail=f"YOLO 不可用：{message}")
    if not (vision.DATASETS_DIR / dataset).is_dir():
        raise HTTPException(status_code=404, detail=f"数据集不存在：{dataset}")
    if epochs < 1 or epochs > 1000:
        raise HTTPException(status_code=400, detail="epochs 需在 1-1000 之间")
    job_id = vision.start_train_job(
        dataset=dataset, weights=weights, epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr0
    )
    return {"job_id": job_id}


# ---------- 任务状态 ----------
@router.get("/api/jobs", summary="任务列表")
def list_jobs() -> dict:
    jobs = sorted(vision.JOBS.values(), key=lambda j: j["id"], reverse=True)
    return {"count": len(jobs), "jobs": jobs}


@router.get("/api/jobs/{job_id}", summary="任务状态")
def get_job(job_id: str) -> dict:
    job = vision.JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


@router.post("/api/jobs/{job_id}/cancel", summary="取消任务")
def cancel_job(job_id: str) -> dict:
    """取消运行中的视频推理 / 训练任务（后台线程会在下一帧/下一轮停下）"""
    if job_id not in vision.JOBS:
        raise HTTPException(status_code=404, detail="任务不存在")
    ok = vision.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="任务已结束，无法取消")
    common.logger.info("已请求取消任务：%s", job_id)
    return {"ok": True, "job_id": job_id}
