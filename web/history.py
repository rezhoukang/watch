"""history: extracted API routes; URLs and response formats are unchanged."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import inference_history
from . import common

router = APIRouter()


def _history_record(record_id: str) -> dict:
    try:
        return inference_history.read_record(record_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="推理记录不存在") from e


@router.get("/api/inferences", summary="历史推理结果图片列表")
def list_inferences() -> dict:
    records = inference_history.list_records()
    return {"count": len(records), "records": records}


@router.post("/api/inferences/delete", summary="批量删除推理记录及其图片")
def delete_inferences(payload: dict) -> dict:
    record_ids = common._delete_items(payload, "record_ids")
    try:
        for record_id in record_ids:
            inference_history.record_dir(record_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return common._delete_batch(record_ids, inference_history.delete_record)


@router.delete("/api/inferences/{record_id}", summary="删除推理记录及其图片")
def delete_inference(record_id: str) -> dict:
    try:
        inference_history.delete_record(record_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="推理记录不存在") from e
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"删除失败：{e}") from e
    return {"ok": True, "record_id": record_id}


@router.post("/api/inferences/merge", summary="合并两次推理（重叠保留基准框）")
def merge_inferences(payload: dict) -> dict:
    base_id, incoming_id = payload.get("base_id"), payload.get("incoming_id")
    if not isinstance(base_id, str) or not isinstance(incoming_id, str):
        raise HTTPException(status_code=400, detail="请选择基准和待合并记录")
    try:
        return inference_history.merge_records(base_id, incoming_id, payload.get("merge_iou", 0.0), payload.get("merge_confidence", 0.0))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="推理记录不存在") from e


@router.get("/api/inferences/{record_id}", summary="查看历史推理详情")
def get_inference(record_id: str) -> dict:
    return _history_record(record_id)


@router.get("/api/inferences/{record_id}/image/{kind}", summary="访问历史推理原图或结果图")
def get_inference_image(record_id: str, kind: str) -> FileResponse:
    record = _history_record(record_id)
    if kind == "source":
        path = inference_history.source_path(record)
    elif kind == "result":
        path = inference_history.record_dir(record_id) / "result.png"
    else:
        raise HTTPException(status_code=400, detail="图片类型只能为 source 或 result")
    return FileResponse(path)
