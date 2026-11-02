"""Fixed single-factor decoding experiments shared by API, jobs and inference."""
from typing import Literal

Experiment = Literal['A', 'B', 'C', 'D']
PRESETS = {
    'A': ('A 基线', 1, 256),
    'B': ('B 束搜索', 2, 256),
    'C': ('C 束搜索', 3, 256),
    'D': ('D 输出预算', 1, 128),
}


def parameters(experiment: str = 'A') -> dict:
    if experiment not in PRESETS:
        raise ValueError('请选择 A、B、C 或 D 解码实验')
    _, beams, tokens = PRESETS[experiment]
    return {'do_sample': False, 'num_beams': beams, 'max_new_tokens': tokens}


def describe(experiment: str = 'A') -> dict:
    values = parameters(experiment)
    return {'experiment': experiment, 'label': PRESETS[experiment][0], **values}
