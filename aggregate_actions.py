"""Head-bound VLM actions for aggregate inference, independent of face/Pose matching."""
import json
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import cv2
from PIL import Image

import vision
from action_recognition import prompts, vlm
from action_recognition.regions import Expansion, estimate, render_crop

Progress = Callable[[dict], None]


def emit(progress: Progress | None, stage: str, message: str, current: int = 0,
         total: int | None = None, **extra) -> None:
    if progress:
        progress({'stage': stage, 'message': message, 'current': current, 'total': total, **extra})


def analyze_heads(image, targets: list[dict], directory: Path | None = None,
                  progress: Progress | None = None) -> dict:
    """Caller reserves the action job lock; each crop belongs to its source head directly."""
    profile = prompts.load()
    profile_id = prompts.identifier(profile)
    expansion = Expansion()
    started = time.perf_counter()
    source = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    context = nullcontext(str(directory)) if directory is not None else tempfile.TemporaryDirectory(prefix='aggregate-vlm-')
    records = []
    shared_error = None
    emit(progress, 'multimodal', '准备 Head 裁剪与 VLM 动作分析', total=len(targets))
    with context as folder:
        root = Path(folder)
        if targets:
            vision.clear_model_cache()
        for index, target in enumerate(targets, 1):
            emit(progress, 'multimodal', f'正在分析 HC{index:04d}（首次可能需要加载模型）', index - 1, len(targets))
            entry = {'target_index': index - 1, 'head_bbox': list(target['bbox'])}
            try:
                if shared_error:
                    raise ValueError(f'模型运行失败后未继续分析：{shared_error}')
                region = estimate(target['bbox'], source.width, source.height, expansion)
                raw, marked = render_crop(source, region, target['bbox'])
                raw_path, marked_path = root / f'HC{index:04d}.png', root / f'HC{index:04d}_target.png'
                raw.save(raw_path)
                marked.save(marked_path)
                result = vlm.analyze(marked_path, vlm.DEFAULT_MODEL, '4bit', profile)
                action = {**result, 'source': 'vlm', 'status': 'done', 'region_bbox': region,
                          'prompt_config_id': profile_id}
                entry.update(action=action, raw_file=raw_path.name, marked_file=marked_path.name)
            except Exception as error:
                action = {'source': 'vlm', 'status': 'failed', 'error': str(error), 'prompt_config_id': profile_id}
                entry['action'] = action
                if isinstance(error, RuntimeError):
                    shared_error = str(error)
            target['action'] = action
            records.append(entry)
            emit(progress, 'multimodal', f'已处理 {index}/{len(targets)} 个目标', index, len(targets),
                 failed=sum(r['action']['status'] == 'failed' for r in records))
        if directory is not None:
            (root / 'record.json').write_text(json.dumps({
                'prompt_config_id': profile_id, 'config': profile, 'compiled_prompt': prompts.compile_prompt(profile),
                'expansion': expansion.to_dict(), 'model': vlm.MODELS[vlm.DEFAULT_MODEL],
                'quantization': '4bit', 'experiment': 'A', 'results': records,
            }, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'enabled': True, 'model': vlm.MODELS[vlm.DEFAULT_MODEL], 'quantization': '4bit',
            'experiment': 'A', 'expansion': expansion.to_dict(), 'prompt_config_id': profile_id,
            'count': len(targets), 'completed': sum(t['action']['status'] == 'done' for t in targets),
            'failed': sum(t['action']['status'] == 'failed' for t in targets),
            'elapsed_seconds': round(time.perf_counter() - started, 3)}
