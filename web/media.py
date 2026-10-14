"""media: extracted API routes; URLs and response formats are unchanged."""
from pathlib import Path
import json
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import vision
from . import common

router = APIRouter()


# ---------- 上传 ----------
@router.post("/api/upload", summary="上传图片")
async def upload_image(file: UploadFile = File(...)) -> dict:
    filename, overwritten = common._save_upload(file, common.UPLOAD_DIR, common.IMAGE_EXTENSIONS)
    common.logger.info("上传图片：%s%s", filename, "（覆盖同名文件）" if overwritten else "")
    return {"filename": filename, "url": f"/images/{filename}", "overwritten": overwritten}


@router.post("/api/upload_video", summary="上传视频")
async def upload_video(file: UploadFile = File(...)) -> dict:
    filename, overwritten = common._save_upload(file, common.VIDEO_DIR, common.VIDEO_EXTENSIONS)
    common.logger.info("上传视频：%s%s", filename, "（覆盖同名文件）" if overwritten else "")
    return {"filename": filename, "url": f"/videos/{filename}", "overwritten": overwritten}


@router.delete("/api/images/{filename}", summary="删除图片")
def delete_image(filename: str) -> dict:
    """删除待测图片（已生成的推理结果图不动）"""
    path = common._safe_file(common.UPLOAD_DIR, filename)
    try:
        path.unlink()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"删除失败：{e}") from e
    common.logger.info("已删除图片：%s", filename)
    return {"ok": True, "filename": filename}


@router.post("/api/images/delete", summary="批量删除待推理图片")
def delete_images(payload: dict) -> dict:
    names = common._delete_items(payload, "filenames")
    # Validate the whole request before changing any files.
    if any("/" in name or "\\" in name or ".." in name or Path(name).name != name for name in names):
        raise HTTPException(status_code=400, detail="非法文件名")
    return common._delete_batch(names, lambda name: (common.UPLOAD_DIR / name).unlink())


# ---------- 列表 ----------
@router.get("/api/images", summary="图片 + 结果列表")
def list_images() -> dict:
    files = (
        sorted((p.name for p in common.UPLOAD_DIR.iterdir() if p.is_file()), reverse=True)
        if common.UPLOAD_DIR.is_dir()
        else []
    )
    results = (
        sorted((p.name for p in common.RESULTS_DIR.iterdir() if p.is_file()), reverse=True)
        if common.RESULTS_DIR.is_dir()
        else []
    )
    return {
        "count": len(files),
        "images": [{"filename": f, "url": f"/images/{f}"} for f in files],
        "results": [{"filename": f, "url": f"/results/{f}"} for f in results],
    }


@router.get("/api/videos", summary="视频 + 输出列表")
def list_videos() -> dict:
    files = (
        sorted((p.name for p in common.VIDEO_DIR.iterdir() if p.is_file()), reverse=True)
        if common.VIDEO_DIR.is_dir()
        else []
    )
    outputs = (
        sorted((p.name for p in common.OUTPUT_DIR.iterdir() if p.is_file()), reverse=True)
        if common.OUTPUT_DIR.is_dir()
        else []
    )
    # 扫描 uploads 里视频检测提取出的「最佳帧」（<视频名>_best_<帧号>.jpg）
    best_frames: dict[str, str] = {}
    if common.UPLOAD_DIR.is_dir():
        for p in common.UPLOAD_DIR.iterdir():
            if p.is_file() and p.suffix.lower() in common.IMAGE_EXTENSIONS and "_best_" in p.name:
                orig = p.name.split("_best_", 1)[0]
                best_frames.setdefault(orig, p.name)
    out_list: list[dict] = []
    frames_dir = vision.OUTPUT_DIR / "frames"
    for f in outputs:
        entry: dict = {"filename": f, "url": f"/outputs/{f}"}
        # 人数折线图数据：<视频输出名>.json 侧车（每帧 head 计数）
        chart_json = frames_dir / f"{Path(f).stem}.json"
        if chart_json.is_file():
            entry["chart_url"] = f"/outputs/frames/{chart_json.name}"
            try:
                metadata = json.loads(chart_json.read_text(encoding="utf-8"))
                entry["head_detection"] = metadata.get("head_detection", False)
                entry["parameters"] = metadata.get("parameters")
                entry["detections_url"] = metadata.get("detections_url")
            except (ValueError, OSError):
                pass
        # 「图预览」：输出视频名去掉 _det... 后即原始视频名，匹配它提取的最佳帧
        orig = f.split("_det", 1)[0]
        bf = best_frames.get(orig)
        if bf:
            entry["best_frame"] = bf
            entry["best_frame_url"] = f"/images/{bf}"
        out_list.append(entry)
    return {
        "count": len(files),
        "videos": [{"filename": f, "url": f"/videos/{f}"} for f in files],
        "outputs": out_list,
    }


@router.get("/api/videos/{filename}/info", summary="视频元信息（时长/分辨率/帧率）")
def video_info(filename: str) -> dict:
    """给前端时间轴用的：总时长（秒）、宽高、帧率、总帧数"""
    path = common._safe_file(common.VIDEO_DIR, filename)
    info = vision.video_info(path)
    if info is None:
        raise HTTPException(status_code=400, detail="无法读取视频信息（文件损坏或格式不支持）")
    return info


@router.delete("/api/videos/{filename}", summary="删除视频")
def delete_video(filename: str) -> dict:
    """删除上传的视频（不删已生成的推理输出）"""
    path = common._safe_file(common.VIDEO_DIR, filename)
    try:
        path.unlink()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"删除失败：{e}") from e
    common.logger.info("已删除视频：%s", filename)
    return {"ok": True, "filename": filename}


# ---------- 文件访问 ----------
@router.get("/images/{filename}", summary="访问图片")
def get_image(filename: str) -> FileResponse:
    return FileResponse(common._safe_file(common.UPLOAD_DIR, filename))


@router.get("/results/{filename}", summary="访问推理结果图")
def get_result(filename: str) -> FileResponse:
    return FileResponse(common._safe_file(common.RESULTS_DIR, filename))


@router.get("/videos/{filename}", summary="访问视频")
def get_video(filename: str) -> FileResponse:
    return FileResponse(common._safe_file(common.VIDEO_DIR, filename))


@router.get("/outputs/frames/{filename}", summary="访问视频折线图数据（每帧人数）")
def get_output_frame_data(filename: str) -> FileResponse:
    """访问 data/outputs/frames/ 下的折线图 json（head_counts）"""
    return FileResponse(common._safe_file(vision.OUTPUT_DIR / "frames", filename))


@router.get("/outputs/{filename}", summary="访问推理结果视频")
def get_output(filename: str) -> FileResponse:
    return FileResponse(common._safe_file(common.OUTPUT_DIR, filename))


@router.get("/outputs/detections/{filename}", summary="视频融合逐帧坐标及类别")
def get_output_detections(filename: str) -> FileResponse:
    return FileResponse(common._safe_file(common.OUTPUT_DIR / "detections", filename),
                        media_type="application/x-ndjson", filename=filename)


@router.get("/models/{run}/{filename}", summary="下载训练得到的权重")
def get_model_file(run: str, filename: str) -> FileResponse:
    if "/" in run or "\\" in run or ".." in run:
        raise HTTPException(status_code=400, detail="非法路径")
    return FileResponse(common._safe_file(vision.RUNS_DIR / run / "weights", filename))


@router.delete("/api/outputs/{filename}", summary="删除推理结果视频")
def delete_output(filename: str) -> dict:
    """删除 data/outputs/ 里的推理结果视频（连同折线图数据侧车 json）"""
    path = common._safe_file(common.OUTPUT_DIR, filename)
    try:
        path.unlink()
        # 折线图数据侧车（<视频stem>.json）
        side = vision.OUTPUT_DIR / "frames" / f"{Path(filename).stem}.json"
        side.unlink(missing_ok=True)
        (common.OUTPUT_DIR / "detections" / f"{Path(filename).stem}.jsonl").unlink(missing_ok=True)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"删除失败：{e}") from e
    common.logger.info("已删除推理结果：%s", filename)
    return {"ok": True, "filename": filename}
