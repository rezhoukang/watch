"""detection: extracted API routes; URLs and response formats are unchanged."""
import time
from fastapi import APIRouter, HTTPException, Query
import vision
import inference_history
import head_fusion
import rtdetrv4
from . import common

router = APIRouter()


# ---------- 推理 ----------
@router.post("/api/detect/image/{filename}", summary="推理图片")
def detect_image(
    filename: str,
    conf: float | None = None,
    weights: str = vision.DEFAULT_WEIGHTS,
    classes: str | None = None,
    imgsz: int = 640,
    head_detection: bool = False,
    rtdetr_base: bool = False,
    merge_iou: float = Query(default=head_fusion.DEFAULT_MERGE_IOU, ge=0, le=1),
    merge_confidence: float = Query(default=head_fusion.DEFAULT_MERGE_CONFIDENCE, ge=0, le=1),
    head_down_r_threshold: float = Query(default=0.05, ge=-1, le=1),
    head_up_r_threshold: float = Query(default=0.20, ge=-1, le=1),
    pose_roi: bool = True,
    roi_confidence: float = Query(default=vision.DEFAULT_ROI_CONFIDENCE, ge=0, le=1),
) -> dict:
    common._validate_head_thresholds(head_down_r_threshold, head_up_r_threshold)
    if head_detection:
        common._require_head_detection()
        weights, conf, imgsz, classes = head_fusion.BASE_WEIGHTS, head_fusion.BASE_CONFIDENCE, head_fusion.IMAGE_SIZE, None
    else:
        common._require_image_backend(weights)

    conf = common._clamp_conf(vision.resolve_confidence(weights, conf))
    class_list = common._parse_classes(classes)

    src = common._safe_file(common.UPLOAD_DIR, filename)
    common.logger.info(
        "开始推理图片：%s（conf=%s, weights=%s, imgsz=%s, classes=%s）",
        filename,
        conf,
        weights,
        imgsz,
        class_list,
    )
    t0 = time.perf_counter()

    try:
        record_id, snapshot = inference_history.begin_record(src)
        result = head_fusion.detect_image(snapshot, rtdetr_base=rtdetr_base, merge_iou=merge_iou, merge_confidence=merge_confidence) if head_detection else vision.detect_image(
            snapshot, weights=weights, conf=conf, classes=class_list, imgsz=imgsz, save_result=False, head_down_r_threshold=head_down_r_threshold, head_up_r_threshold=head_up_r_threshold, pose_roi=pose_roi, roi_confidence=roi_confidence
        )
        result["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        result = inference_history.complete_record(
            record_id, snapshot, result,
            head_fusion.parameters(rtdetr_base, merge_iou, merge_confidence) if head_detection else
            {"confidence": conf, "weights": weights, "classes": class_list, "imgsz": imgsz, "head_down_r_threshold": head_down_r_threshold, "head_up_r_threshold": head_up_r_threshold, "pose_roi": pose_roi, "roi_confidence": roi_confidence},
            kind="merge" if head_detection else "inference",
        )
    except Exception as e:  # noqa: BLE001
        common.logger.exception("推理图片失败：%s", filename)
        raise HTTPException(
            status_code=500, detail=f"推理失败（{type(e).__name__}）：{e}"
        ) from e

    elapsed = time.perf_counter() - t0
    common.logger.info(
        "推理完成：%s 模式=%s 检出 %s 个目标，耗时 %.2fs",
        filename,
        result.get("mode"),
        result.get("count"),
        elapsed,
    )
    return result


@router.post("/api/detect/video/{filename}", summary="推理视频（后台任务）")
def detect_video(
    filename: str,
    conf: float | None = None,
    stride: int = 1,
    weights: str = vision.DEFAULT_WEIGHTS,
    classes: str | None = None,
    start: float = 0.0,
    end: float | None = None,
    render: str = "box",
    line_width: int = 2,
    head_detection: bool = False,
    rtdetr_base: bool = False,
    merge_iou: float = Query(default=head_fusion.DEFAULT_MERGE_IOU, ge=0, le=1),
    merge_confidence: float = Query(default=head_fusion.DEFAULT_MERGE_CONFIDENCE, ge=0, le=1),
    head_down_r_threshold: float = Query(default=0.05, ge=-1, le=1),
    head_up_r_threshold: float = Query(default=0.20, ge=-1, le=1),
    pose_roi: bool = True,
    roi_confidence: float = Query(default=vision.DEFAULT_ROI_CONFIDENCE, ge=0, le=1),
) -> dict:
    common._validate_head_thresholds(head_down_r_threshold, head_up_r_threshold)
    if head_detection:
        common._require_head_detection()
        weights, conf, classes = head_fusion.BASE_WEIGHTS, head_fusion.BASE_CONFIDENCE, None
    common._require_image_backend(weights)

    src = common._safe_file(common.VIDEO_DIR, filename)
    stride = max(1, min(stride, 30))
    conf = common._clamp_conf(vision.resolve_confidence(weights, conf))
    class_list = common._parse_classes(classes)

    # 显示形式：'box' 框 / 'dot' 中心点；line_width 钳到 1~8
    if render not in ("box", "dot"):
        raise HTTPException(status_code=400, detail="render 只能是 box 或 dot")
    line_width = max(1, min(line_width, 8))

    # 时间范围校验：end 缺省 = 全片；必须 start < end
    start = max(0.0, start)
    if end is not None and end <= start:
        raise HTTPException(status_code=400, detail="时间范围无效：end 必须大于 start")

    try:
        job_id = vision.start_video_job(
            src,
            weights=weights,
            conf=conf,
            stride=stride,
            classes=class_list,
            start=start,
            end=end,
            render=render,
            line_width=line_width,
            head_detection=head_detection,
            rtdetr_base=rtdetr_base,
            merge_iou=merge_iou,
            merge_confidence=merge_confidence,
            head_down_r_threshold=head_down_r_threshold, head_up_r_threshold=head_up_r_threshold, pose_roi=pose_roi, roi_confidence=roi_confidence,
        )
    except Exception as e:  # noqa: BLE001
        common.logger.exception("提交视频推理任务失败：%s", filename)
        raise HTTPException(
            status_code=500, detail=f"提交失败（{type(e).__name__}）：{e}"
        ) from e

    common.logger.info(
        "已提交视频推理任务：%s（job=%s, weights=%s, conf=%s, stride=%s, classes=%s, "
        "range=%s~%ss, render=%s, lw=%s）",
        filename,
        job_id,
        weights,
        conf,
        stride,
        class_list,
        start,
        end,
        render,
        line_width,
    )
    return {"job_id": job_id, "head_detection": head_detection, "rtdetr_base": head_detection and rtdetr_base,
            "merge_iou": merge_iou if head_detection else 0.0,
            "merge_confidence": merge_confidence if head_detection else 0.0}
