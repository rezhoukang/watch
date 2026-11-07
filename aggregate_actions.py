"""Aggregate VLM strategy one: all Head results, then all context results."""
import copy
import json
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import cv2
from PIL import Image

import vision
from action_recognition import head_only, prompts, two_stage, vlm
from action_recognition.regions import Expansion, estimate, render_crop

Progress = Callable[[dict], None]


def emit(progress: Progress | None, stage: str, message: str, current: int = 0,
         total: int | None = None, **extra) -> None:
    if progress:
        progress({'stage': stage, 'message': message, 'current': current, 'total': total, **extra})


def analyze_heads(image, targets: list[dict], directory: Path | None = None,
                  progress: Progress | None = None) -> dict:
    profile = head_only.profile()
    context_profile = two_stage.context_profile(prompts.load())
    profile_id = prompts.identifier(profile)
    expansion = Expansion()
    started = time.perf_counter()
    source = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    context = nullcontext(str(directory)) if directory is not None else tempfile.TemporaryDirectory(prefix='aggregate-vlm-')
    records, entries = [], []
    with context as folder:
        root = Path(folder)
        for index, target in enumerate(targets, 1):
            region = estimate(target['bbox'], source.width, source.height, expansion)
            raw, marked = render_crop(source, region, target['bbox'])
            raw_file, context_file, head_file = f'HC{index:04d}.png', f'HC{index:04d}_target.png', f'HC{index:04d}_head.png'
            raw.save(root / raw_file)
            marked.save(root / context_file)
            x1, y1, x2, y2 = target['bbox']
            source.crop((x1, y1, x2, y2)).save(root / head_file)
            entries.append({'target': target, 'target_index': index - 1, 'head_bbox': list(target['bbox']),
                            'context_bbox': region, 'head_file': head_file, 'context_file': context_file,
                            'head_path': root / head_file, 'context_path': root / context_file})
            records.append({'target_index': index - 1, 'head_bbox': list(target['bbox']),
                            'raw_file': raw_file, 'marked_file': context_file, 'head_file': head_file})

        def persist():
            if directory is not None:
                manifest = {'prompt_config_id': profile_id, 'config': profile,
                            'compiled_prompt': prompts.compile_prompt(profile),
                            'context_config': context_profile, 'context_prompt_config_id': prompts.identifier(context_profile),
                            'context_compiled_prompt': prompts.compile_prompt(context_profile),
                            'two_stage': True, 'expansion': expansion.to_dict(),
                            'model': vlm.MODELS[vlm.DEFAULT_MODEL], 'quantization': '4bit',
                            'experiment': 'A', 'results': records}
                (root / 'record.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

        def report(phase, current, total):
            label = '第一阶段：全部 Head 抬头/低头' if phase == 'head' else '第二阶段：全部扩展 ROI 补充'
            emit(progress, 'multimodal', f'{label} {current}/{total}',
                 current + (total if phase == 'context' else 0), 2 * total,
                 pass_name=phase, pass_current=current, pass_total=total,
                 preview_targets=copy.deepcopy(targets))

        def publish(entry, phase, result, error):
            if error:
                action = {**(entry['target'].get('action') or {}), 'source': 'vlm',
                          'status': 'failed', 'error': str(error), 'two_stage': True}
            else:
                action = {**result, 'source': 'vlm', 'status': 'pending_context' if phase == 'head' else 'done',
                          'provisional': phase == 'head', 'two_stage': True,
                          'region_bbox': entry[f'{phase}_bbox'], 'prompt_config_id': profile_id}
            entry['target']['action'] = action
            records[entry['target_index']]['action'] = action
            persist()

        if entries:
            vision.clear_model_cache()
        report('head', 0, len(entries))
        two_stage.run_batch(entries, vlm.DEFAULT_MODEL, '4bit', profile, context_profile, 'A', report, publish)
        persist()
    return {'enabled': True, 'two_stage': True, 'model': vlm.MODELS[vlm.DEFAULT_MODEL], 'quantization': '4bit',
            'experiment': 'A', 'expansion': expansion.to_dict(), 'prompt_config_id': profile_id,
            'count': len(targets), 'completed': sum(t.get('action', {}).get('status') == 'done' for t in targets),
            'failed': sum(t.get('action', {}).get('status') == 'failed' for t in targets),
            'elapsed_seconds': round(time.perf_counter() - started, 3)}
