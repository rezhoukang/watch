"""datasets: extracted API routes; URLs and response formats are unchanged."""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
import vision
import head_fusion
from . import common

router = APIRouter()


# ---------- 训练 ----------
@router.get("/api/datasets", summary="数据集列表")
def list_datasets() -> dict:
    datasets = vision.list_datasets()
    return {"count": len(datasets), "datasets": datasets}


@router.get("/api/datasets/{name}", summary="数据集详情（含 Train/Val 子集与图片列表）")
def dataset_detail(name: str) -> dict:
    ds_dir = vision.DATASETS_DIR / name
    if not ds_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"数据集不存在：{name}")
    return vision.dataset_detail(ds_dir)


@router.put("/api/datasets/{name}/classes", summary="改写数据集类别名")
def write_dataset_classes(name: str, payload: dict) -> dict:
    """body: {"classes": ["person", "flag", ...]}

    只改名字，class_id（行号）不变，所以已有 label 文件不用重写；
    后端会一并同步 data.yaml 的 nc/names 并删掉训练标注缓存。
    """
    ds_dir = vision.DATASETS_DIR / name
    if not ds_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"数据集不存在：{name}")
    names = payload.get("classes")
    if not isinstance(names, list):
        raise HTTPException(status_code=400, detail="classes 必须是数组")
    try:
        saved = vision.write_dataset_classes(ds_dir, names)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    common.logger.info("数据集 %s 的类别已更新：%s", name, saved)
    return {"ok": True, "classes": saved}


# ---------- 标注（训练数据准备）----------
# 注意：filename 形如 <数据集>/<split>/<图片名>，含斜杠，所以用 {filename:path}
# 与推理共用同一套权重选项（conf/weights/classes/imgsz）。
@router.get("/api/annotate/image/{filename:path}", summary="标注器：取一张图片的现有标注")
def annotate_image(filename: str) -> dict:
    """filename 形如 <dataset>/<split>/<img>，如 demo/train/abc.jpg
    返回该图片已有标注（YOLO 坐标已还原为原图像素）+ 类别清单。
    """
    img_path = vision.resolve_annotate_image(filename)
    labels = vision.read_labels_for_image(img_path)
    ds_dir = vision.find_dataset_dir(img_path)
    classes = vision.dataset_classes(ds_dir)
    return {
        "ok": True,
        "dataset": ds_dir.name,
        "split": vision.split_of(img_path),
        "filename": img_path.name,
        "url": f"/api/annotate/raw/{filename}",
        "width": labels["width"],
        "height": labels["height"],
        "detections": labels["detections"],
        "classes": classes,
    }


@router.get("/api/annotate/raw/{filename:path}", summary="访问标注器里的原图")
def get_annotate_image(filename: str):
    return FileResponse(vision.resolve_annotate_image(filename))


@router.post("/api/annotate/prelabel/{filename:path}", summary="标注器：对当前图片做预标注")
def prelabel_image(
    filename: str,
    conf: float = 0.38,
    weights: str = vision.DEFAULT_WEIGHTS,
    classes: str | None = None,
    imgsz: int = 640,
    head_detection: bool = False,
    rtdetr_base: bool = False,
    merge_iou: float = Query(default=head_fusion.DEFAULT_MERGE_IOU, ge=0, le=1),
    merge_confidence: float = Query(default=head_fusion.DEFAULT_MERGE_CONFIDENCE, ge=0, le=1),
) -> dict:
    """用与推理一致的权重选项对单张图片做推理，结果转成标注框返回。
    filename 形如 <dataset>/<split>/<img>。
    """
    if head_detection:
        common._require_head_detection()
        weights, conf, imgsz, classes = head_fusion.BASE_WEIGHTS, head_fusion.BASE_CONFIDENCE, head_fusion.IMAGE_SIZE, None
    common._require_yolo_weights(weights)
    img_path = vision.resolve_annotate_image(filename)
    available, message = vision.yolo_available()
    if not available:
        raise HTTPException(status_code=503, detail=f"YOLO 不可用：{message}")
    ok, reason = vision.world_ready(weights)
    if not ok:
        raise HTTPException(status_code=503, detail=reason)

    conf = common._clamp_conf(conf)
    class_list = common._parse_classes(classes)
    ds_dir = vision.find_dataset_dir(img_path)
    ds_classes = vision.dataset_classes(ds_dir)

    try:
        fused = head_fusion.detect_image(img_path, rtdetr_base=rtdetr_base, merge_iou=merge_iou, merge_confidence=merge_confidence) if head_detection else None
        dets = fused["detections"] if fused is not None else vision.prelabel_image(
            img_path, weights=weights, conf=conf, classes=class_list, imgsz=imgsz
        )
    except Exception as e:  # noqa: BLE001
        common.logger.exception("预标注失败：%s", filename)
        raise HTTPException(
            status_code=500, detail=f"预标注失败（{type(e).__name__}）：{e}"
        ) from e

    # 模型输出 → 数据集类别：命中就用；未命中则追加为新类别（不重排、不影响已有 id）
    unknown = [d["class_name"] for d in dets if d["class_name"] not in ds_classes]
    if unknown:
        ds_classes = vision.append_dataset_classes(ds_dir, unknown)
        common.logger.info("预标注新增类别：%s", sorted(set(unknown)))
    ds_name_to_id = {n: i for i, n in enumerate(ds_classes)}
    mapped = []
    for d in dets:
        cid = ds_name_to_id.get(d["class_name"])
        if cid is None:
            cid = 0
            d["class_name"] = ds_classes[0] if ds_classes else str(0)
        d["class_id"] = cid
        mapped.append(d)

    common.logger.info(
        "预标注完成：%s（conf=%s, weights=%s, 检出 %s 个）",
        filename,
        conf,
        weights,
        len(mapped),
    )
    return {"ok": True, "detections": mapped, "classes": ds_classes,
            "head_detection": head_detection, "fusion": fused["fusion"] if fused else None,
            "parameters": head_fusion.parameters(rtdetr_base, merge_iou, merge_confidence) if head_detection else None}


@router.post("/api/annotate/save/{filename:path}", summary="标注器：保存标注到 YOLO label 文件")
def annotate_save(filename: str, payload: dict) -> dict:
    """body: {"detections": [{"class_id", "bbox": [x1,y1,x2,y2]}, ...]}
    保存为 <数据集>/labels/<split>/<图名>.txt（YOLO 归一化格式）。
    """
    img_path = vision.resolve_annotate_image(filename)
    dets = payload.get("detections", [])
    if not isinstance(dets, list):
        raise HTTPException(status_code=400, detail="detections 必须是数组")
    saved, lbl_rel = vision.save_labels_for_image(img_path, dets)
    common.logger.info("保存标注：%s → %s（%s 个目标）", filename, lbl_rel, saved)
    return {"ok": True, "count": saved, "label": lbl_rel}
