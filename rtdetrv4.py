"""RT-DETRv4-S C-RADIOv4 student: head-only classroom image counting.

Uses the official RT-DETRv4 engine, not Ultralytics RTDETR. Heavy imports and
model construction are lazy so the web app can start without this backend.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_NAME = "best_stg2.safetensors"
MODEL_NAME = "RT-DETRv4-S C-RADIOv4（人头计数）"
HEAD_CLASS_ID = 1  # Release class order: person, head. Do not count both.
REPO_REVISION = "55fefaaed7efe2a5f72d0a18fd4e05965e35c292"
WEIGHTS_REVISION = "823ca54bbf1446173f6906198a5e8b2a3e090890"
WEIGHTS_SHA256 = "8a0cc9acdf6adccf39511ac626335c1deb0003278e74b12295d70cdb6a44c869"
_lock = threading.RLock()
_detector: Any = None


def is_weights(weights: str) -> bool:
    return Path(weights).name == WEIGHTS_NAME


def repo_dir() -> Path:
    return Path(os.environ.get("RTDETRV4_REPO", BASE_DIR / "third_party" / "RT-DETRv4")).resolve()


def status() -> dict:
    """Report installation readiness without loading a model or downloading files."""
    missing = []
    if not (BASE_DIR / "models" / WEIGHTS_NAME).is_file():
        missing.append(f"缺少 models/{WEIGHTS_NAME}")
    if not (repo_dir() / "engine" / "core" / "yaml_config.py").is_file():
        missing.append("缺少官方源码，请运行 python scripts/setup_rtdetrv4.py")
    packages = {
        "torch": "torch", "torchvision": "torchvision", "safetensors": "safetensors",
        "PIL": "pillow", "numpy": "numpy", "yaml": "PyYAML", "scipy": "scipy",
        "faster_coco_eval": "faster-coco-eval", "calflops": "calflops",
        "tensorboard": "tensorboard", "transformers": "transformers",
    }
    absent = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
    if absent:
        missing.append("缺少依赖 " + ", ".join(absent) + "；请运行 pip install -r requirements-rtdetrv4.txt")
    return {
        "available": not missing,
        "message": "；".join(missing) if missing else f"{MODEL_NAME} 已安装，首次推理加载模型",
        "weight": WEIGHTS_NAME,
        "model_name": MODEL_NAME,
        "classes": ["head"],
        "head_class_id": HEAD_CLASS_ID,
    }


class _HeadDetector:
    def __init__(self) -> None:
        import torch
        from safetensors.torch import load_file

        repo = repo_dir()
        # 'engine' is a generic upstream package name; reject another engine
        # already imported rather than silently constructing the wrong model.
        engine = sys.modules.get("engine")
        if engine is not None and Path(engine.__file__).resolve().parent != repo / "engine":
            raise RuntimeError("已加载其他 engine 包，请在独立进程中运行 RT-DETRv4")
        sys.path.insert(0, str(repo))
        try:
            from engine.core import YAMLConfig
        finally:
            sys.path.remove(str(repo))

        cfg = YAMLConfig(
            str(repo / "configs" / "rtv4" / "rtv4_hgnetv2_s_coco.yml"),
            num_classes=2,
            remap_mscoco_category=False,
            HGNetv2={"pretrained": False},
            HybridEncoder={"distill_teacher_dim": 1152},
        )
        state = load_file(str(BASE_DIR / "models" / WEIGHTS_NAME), device="cpu")
        model = cfg.model
        # Strict validation catches a wrong architecture/checkpoint immediately.
        model.load_state_dict(state, strict=True)
        # The release stores 640x640 anchors. After loading, generate anchors and
        # positional encodings dynamically to support the existing imgsz control.
        model.encoder.eval_spatial_size = None
        model.decoder.eval_spatial_size = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model.deploy().to(self.device).eval()
        self.postprocessor = cfg.postprocessor.deploy().to(self.device).eval()

    def predict(self, image: Any, imgsz: int) -> tuple[Any, Any, Any]:
        import torch
        from torchvision.transforms import functional as F

        # Match official inference: RGB, square resize, float [0,1]; no letterbox
        # or ImageNet normalization. orig_target_sizes is [width, height].
        resized = F.resize(image, [imgsz, imgsz])
        tensor = F.to_tensor(resized).unsqueeze(0).to(self.device)
        orig_size = torch.tensor([list(image.size)], device=self.device)
        with torch.inference_mode():
            labels, boxes, scores = self.postprocessor(self.model(tensor), orig_size)
        return labels[0].cpu().tolist(), boxes[0].cpu().tolist(), scores[0].cpu().tolist()


def clear_cache() -> int:
    global _detector
    with _lock:
        count = int(_detector is not None)
        _detector = None
        return count


def _head_detections(labels, boxes, scores, width: int, height: int, conf: float) -> list[dict]:
    detections = []
    for label, box, score in zip(labels, boxes, scores):
        if label != HEAD_CLASS_ID or not math.isfinite(score) or score < conf:
            continue
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            continue
        x1, y1, x2, y2 = [
            round(max(0.0, min(float(v), limit)), 2)
            for v, limit in zip(box, [width, height, width, height])
        ]
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            "class_id": HEAD_CLASS_ID,
            "class_name": "head",
            "confidence": round(float(score), 4),
            "bbox": [x1, y1, x2, y2],
            "bbox_cxcywh": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2),
                            round(x2 - x1, 2), round(y2 - y1, 2)],
        })
    return detections


def detect_image(src: Path, conf: float, imgsz: int, results_dir: Path, save_result: bool = True) -> dict:
    global _detector
    from PIL import Image, ImageDraw

    with Image.open(src) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    with _lock:
        if _detector is None:
            ready = status()
            if not ready["available"]:
                raise RuntimeError(ready["message"])
            _detector = _HeadDetector()
        labels, boxes, scores = _detector.predict(image, imgsz)
        device = _detector.device
    detections = _head_detections(labels, boxes, scores, width, height, conf)

    out_name = ""
    if save_result:
        results_dir.mkdir(parents=True, exist_ok=True)
        # Avoid overwriting YOLO's <stem>_det output for the same classroom image.
        out_name = f"{src.stem}_rtdetrv4_head.jpg"
        draw = ImageDraw.Draw(image)
        for det in detections:
            draw.rectangle(det["bbox"], outline="#00dd66", width=2)
        draw.rectangle((0, 0, min(width, 225), min(height, 28)), fill="black")
        draw.text((6, 6), f"Head count: {len(detections)}", fill="white")
        image.save(results_dir / out_name, quality=95)
    return {
        "source": src.name,
        "result": out_name,
        "result_url": f"/results/{out_name}" if out_name else "",
        "image_width": width,
        "image_height": height,
        "imgsz": imgsz,
        "count": len(detections),
        "head_count": len(detections),
        "person_count": len(detections),
        "summary": {"head": len(detections)},
        "detections": detections,
        "mode": "rtdetrv4",
        "weight": WEIGHTS_NAME,
        "model_name": MODEL_NAME,
        "classes_used": ["head"],
        "device": device,
    }
