"""Head-only orientation inputs, isolated from full-context action analysis."""
import copy
import json
from pathlib import Path

from PIL import Image

from . import prompts, store


def _initial(config: dict) -> dict:
    rules = {item['name']: item for item in config['actions']}
    fallback = {item['name']: item for item in prompts.defaults()['actions']}
    actions = []
    for name in ('抬头', '低头'):
        action = copy.deepcopy(rules.get(name, fallback[name]))
        action['enabled'] = True
        actions.append(action)
    return prompts.normalize({
        'version': 1,
        'instructions': '本次图片仅包含目标 Head 框，整张图都是目标头部，不要求红框。只判断抬头、低头或无法判断，不识别其他动作。依据整体面部和头部朝向判断；面向前方或上方包括正常平视，面部明显朝下才支持低头。头发遮脸或眼睛不可见本身不是低头依据。缺少桌面、手部和身体属于本模式正常裁剪，不应据此猜测动作。',
        'actions': actions,
    })


def normalize(config: dict) -> dict:
    config = prompts.normalize(config)
    if (len(config['actions']) != 2 or
            {a['name'] for a in config['actions']} != {'抬头', '低头'} or
            not all(a['enabled'] for a in config['actions']) or config.get('head_only_short')):
        raise ValueError('Head 模式固定启用抬头、低头，请编辑说明、规则和优先级')
    return config


def defaults() -> dict:
    path = Path(__file__).with_name('default_head_prompt.json')
    return normalize(json.loads(path.read_text(encoding='utf-8')))


def save(config: dict) -> dict:
    config = normalize(config)
    with prompts._lock:
        store.ROOT.mkdir(parents=True, exist_ok=True)
        path = store.ROOT / 'head_prompt_config.json'
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(path)
    return config


def profile(config: dict | None = None) -> dict:
    with prompts._lock:
        path = store.ROOT / 'head_prompt_config.json'
        if path.is_file():
            return normalize(json.loads(path.read_text(encoding='utf-8')))
        # One-time migration preserves current rules; later edits are independent.
        return save(_initial(prompts.load() if config is None else config))


def describe(config: dict | None = None) -> dict:
    value = profile() if config is None else normalize(config)
    return {**prompts.describe(value), 'defaults': defaults()}


def input_file(record: dict, item: dict):
    directory = store.folder(record['run_id'])
    relative = f"crops/{item['crop_id']}_head.png"
    with Image.open(directory / item['source_file']) as image:
        x1, y1, x2, y2 = item['head_bbox']
        image.convert('RGB').crop((x1, y1, x2, y2)).save(directory / relative)
    return directory / relative, relative
