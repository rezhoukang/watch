"""Dataset metadata, class lists and YOLO label persistence. No model loading."""
from pathlib import Path

from project_paths import DATASETS_DIR, IMAGE_EXTENSIONS

# ---------- 训练（后台任务）----------
def list_datasets() -> list[dict]:
    """列出 data/datasets/ 下的数据集及其基本信息"""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for d in sorted(DATASETS_DIR.iterdir()):
        if not d.is_dir():
            continue
        # 统计图片数量（支持 images/train 或 images/ 两种结构）
        img_count = 0
        for sub in ("images/train", "images/val", "images"):
            p = d / sub
            if p.is_dir():
                img_count += sum(
                    1 for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
                )
        out.append(
            {
                "name": d.name,
                "images": img_count,
                "has_yaml": (d / "data.yaml").exists(),
            }
        )
    return out


# ---------- 标注（训练数据准备）----------
def dataset_detail(ds_dir: Path) -> dict:
    """数据集的 Train/Val 子集详情：图片清单 + 标注统计"""
    names = dataset_classes(ds_dir)
    out = {
        "name": ds_dir.name,
        "classes": names,
        "subsets": {},
    }
    for split in ("train", "val"):
        img_dir = ds_dir / "images" / split
        if not img_dir.is_dir():
            continue
        files = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        label_dir = ds_dir / "labels" / split
        labeled = sum(
            1 for f in files if (label_dir / Path(f).with_suffix(".txt").name).exists()
        )
        out["subsets"][split] = {
            "count": len(files),
            "labeled": labeled,
            "files": files,
        }
    return out


def resolve_annotate_image(filename: str) -> Path:
    """把 <数据集>/<split>/<图片名> 解析为数据集里的绝对路径（防目录穿越）

    图片所在目录决定 split 与标签目录。
    """
    parts = [p for p in Path(filename).parts if p not in ("", ".", "..")]
    if len(parts) < 2:
        raise ValueError("图片路径格式应为 <数据集>/<split>/<图片名>")
    ds_name = parts[0]
    img_name = parts[-1]
    ds_dir = DATASETS_DIR / ds_name
    if not ds_dir.is_dir():
        raise ValueError(f"数据集不存在：{ds_name}")
    # 只允许 images/{train,val} 下的图片
    candidates = []
    for split in ("train", "val"):
        p = ds_dir / "images" / split / img_name
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            candidates.append(p)
    if not candidates:
        raise ValueError(f"图片不存在：{filename}")
    return candidates[0]


def split_of(img_path: Path) -> str:
    """由图片路径反推 split（train/val）"""
    try:
        return img_path.parent.name
    except Exception:  # noqa: BLE001
        return "train"


def find_dataset_dir(img_path: Path) -> Path:
    """由图片路径向上找到数据集根目录（含 data.yaml / classes.txt 的那层）"""
    d = img_path.parent
    while d != d.parent:
        if (d / "images").is_dir() or (d / "labels").is_dir():
            return d
        d = d.parent
    return DATASETS_DIR / img_path.parts[-3]


def dataset_classes(ds_dir: Path) -> list[str]:
    """数据集的类别清单

    **唯一来源是 data.yaml 的 names**（项目已取消 classes.txt：
    类别只能在图形界面里增删改，改完直接落到 data.yaml）。
    没有 data.yaml 时从 label 反推最大 class_id 兜底。
    """
    names = _read_yaml_names(ds_dir)
    if names:
        return names
    return _infer_classes(ds_dir)


def _read_yaml_names(ds_dir: Path) -> list[str] | None:
    """读 data.yaml 的 names（list / dict 两种写法都支持）。读不到返回 None"""
    yaml_path = ds_dir / "data.yaml"
    if not yaml_path.is_file():
        return None
    try:
        import yaml

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        return _names_of(data)
    except Exception:  # noqa: BLE001
        return None


def _names_of(data: object) -> list[str] | None:
    """把 data.yaml 里的 names 归一化成 list[str]（dict 键按数值排序还原顺序）"""
    if not isinstance(data, dict):
        return None
    raw = data.get("names")
    if isinstance(raw, dict):
        try:
            pairs = sorted(raw.items(), key=lambda kv: int(kv[0]))
        except (TypeError, ValueError):
            pairs = list(raw.items())
        out = [str(v) for _, v in pairs]
        return out or None
    if isinstance(raw, list):
        out = [str(v) for v in raw]
        return out or None
    return None


def _infer_classes(ds_dir: Path) -> list[str]:
    """没有 data.yaml 时，从 label 里反推最大 class_id 生成 class_0, class_1…"""
    max_id = -1
    for lbl_dir in (
        ds_dir / "labels" / "train",
        ds_dir / "labels" / "val",
        ds_dir / "labels",
    ):
        if not lbl_dir.is_dir():
            continue
        for f in lbl_dir.glob("*.txt"):
            for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = ln.split()
                if parts and parts[0].lstrip("-").isdigit():
                    max_id = max(max_id, int(parts[0]))
    return [f"class_{i}" for i in range(max_id + 1)] if max_id >= 0 else ["object"]


def append_dataset_classes(ds_dir: Path, names: list[str]) -> list[str]:
    """把新类别追加到 data.yaml 末尾（不改动已有类别的 id），返回更新后的清单

    预标注时模型可能检出数据集里还没有的类别（如 COCO 的 chair）。
    只追加不重排，保证已有 label 的 class_id 不会错位。
    names 内部的重复项会去重（同一张图可能有多个同类目标）。
    """
    merged = dataset_classes(ds_dir)
    added = False
    for n in names:
        if n and n not in merged:
            merged.append(n)
            added = True
    if added:
        write_dataset_classes(ds_dir, merged)
    return merged


def write_dataset_classes(ds_dir: Path, names: list[str]) -> list[str]:
    """把类别清单写进 data.yaml（names 与 nc 一起写），并清掉训练用的标注缓存

    - data.yaml 是类别的**唯一来源**（项目已取消 classes.txt）
    - 只改名字/追加，class_id 就是下标，所以已有 label 文件不必重写
    - `nc` 必须等于 names 长度：不一致时 ultralytics 会直接抛 SyntaxError 拒绝训练，
      所以两者永远一起写；顺带补齐 path / train / val（用户手写的 yaml 可能缺）
    - 内容已经完全一致时不重写文件、也不清缓存（避免每次训练都全量重扫 label）
    """
    cleaned = [str(n).strip() for n in names]
    if not cleaned or any(not n for n in cleaned):
        raise ValueError("类别名不能为空")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("类别名不能重复")

    yaml_path = ds_dir / "data.yaml"
    data: dict = {}
    if yaml_path.is_file():
        try:
            import yaml

            loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                data = loaded
        except Exception:  # noqa: BLE001
            data = {}   # yaml 坏了就重建一份

    # 已经完全一致 → 什么都不做
    if (
        data.get("nc") == len(cleaned)
        and _names_of(data) == cleaned
        and data.get("path")
        and data.get("train")
        and data.get("val")
    ):
        return cleaned

    # 补齐 ultralytics 需要的字段（只填缺失的，不动用户已有配置）
    data.setdefault("path", ds_dir.as_posix())
    if not data.get("train"):
        data["train"] = "images/train"
    if not data.get("val"):
        data["val"] = (
            "images/val" if (ds_dir / "images" / "val").is_dir() else "images/train"
        )
    data["nc"] = len(cleaned)
    data["names"] = {i: n for i, n in enumerate(cleaned)}

    try:
        import yaml
    except ImportError as e:   # noqa: BLE001
        raise RuntimeError(f"缺少 PyYAML，无法写入 data.yaml：{e}") from e

    # 传文件对象当 stream：返回 None，避免 PyYAML 存根把返回值标成
    # bytes|str|None 而与 write_text 的类型冲突
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    _invalidate_label_cache(ds_dir)
    return cleaned


def _invalidate_label_cache(ds_dir: Path) -> None:
    """删掉 ultralytics 的标注扫描缓存（*.cache）

    必须删：缓存 hash 只由「标签路径列表 + 图片路径列表 + 扫描参数」算出，
    **不含文件内容也不含 mtime**。所以改了标注内容却不删缓存时，
    训练会静默复用旧标签 —— 不报错，但用的是过期数据。
    缓存路径规则见 ultralytics YOLODataset.get_labels：
        Path(label_files[0]).parent.with_suffix(".cache")  →  labels/<split>.cache
    """
    labels_dir = ds_dir / "labels"
    targets = [labels_dir.with_suffix(".cache")]           # labels.cache
    targets += [(labels_dir / s).with_suffix(".cache") for s in ("train", "val")]
    for cache in targets:
        try:
            cache.unlink(missing_ok=True)
        except OSError:
            pass


def read_labels_for_image(img_path: Path) -> dict:
    """读一张图片已有的 YOLO 标注（相对 <数据集>/images/<split>/），还原为原图像素"""
    ds_dir = find_dataset_dir(img_path)
    split = split_of(img_path)
    label_path = ds_dir / "labels" / split / Path(img_path.name).with_suffix(".txt")
    img_w, img_h = _image_size(img_path)
    names = dataset_classes(ds_dir)
    dets = []
    if label_path.is_file():
        for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            dets.append(
                {
                    "class_id": cls,
                    "class_name": names[cls] if cls < len(names) else str(cls),
                    "confidence": None,
                    "bbox": [
                        round((cx - w / 2) * img_w, 2),
                        round((cy - h / 2) * img_h, 2),
                        round((cx + w / 2) * img_w, 2),
                        round((cy + h / 2) * img_h, 2),
                    ],
                }
            )
    return {"width": img_w, "height": img_h, "detections": dets}


def _image_size(img_path: Path) -> tuple[int, int]:
    """读图片宽高（失败回退 1x1，避免除零）

    不能用裸 cv2.imread：Windows 下遇到中文路径会静默返回 None，
    会让标注坐标全部按 1x1 换算而错乱。改用 np.fromfile + cv2.imdecode 读字节。
    """
    try:
        import cv2
        import numpy as np

        buf = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # type: ignore[attr-defined]
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    except Exception:  # noqa: BLE001
        pass
    return 1, 1


def save_labels_for_image(img_path: Path, dets: list[dict]) -> tuple[int, str]:
    """把前端标注写回 YOLO 归一化 label 文件，返回 (目标数, 相对 label 路径)"""
    ds_dir = find_dataset_dir(img_path)
    split = split_of(img_path)
    img_w, img_h = _image_size(img_path)
    if img_w <= 0 or img_h <= 0:
        raise RuntimeError("无法读取图片尺寸，保存失败")

    lines = []
    saved = 0
    for d in dets:
        try:
            cid = int(d.get("class_id", 0))
        except (TypeError, ValueError):
            continue
        bb = d.get("bbox")
        if not isinstance(bb, (list, tuple)) or len(bb) < 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bb[:4])
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        saved += 1

    label_dir = ds_dir / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / Path(img_path.name).with_suffix(".txt")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # 关键：删掉 ultralytics 的标注缓存，否则训练会静默复用旧标签（见函数说明）
    _invalidate_label_cache(ds_dir)
    return saved, str(label_path.relative_to(ds_dir))
