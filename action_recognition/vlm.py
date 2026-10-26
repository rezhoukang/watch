"""Local Qwen3-VL inference, conservative JSON parsing and explicit model readiness."""
import gc
import hashlib
import importlib.util
import json
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from project_paths import MODELS_DIR
from . import prompts

MODELS = {"4b": "Qwen/Qwen3-VL-4B-Instruct", "2b": "Qwen/Qwen3-VL-2B-Instruct"}
DEFAULT_MODEL = "4b"
STATES = tuple(prompts.states(prompts.defaults()))
PROMPT_VERSION = prompts.VERSION
_lock = threading.RLock()
_model: Any = None
_processor: Any = None
_loaded: tuple[str, str] | None = None


def model_path(key: str) -> Path:
    if key not in MODELS:
        raise ValueError("请选择 4B 或 2B 模型")
    return MODELS_DIR / MODELS[key].split("/")[-1]


def weights_ready(key: str) -> bool:
    directory = model_path(key)
    if not all((directory / name).is_file() for name in
               ("config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json")):
        return False
    index = directory / "model.safetensors.index.json"
    try:
        if index.is_file():
            names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            return bool(names) and all((directory / name).is_file() and
                                       (directory / name).stat().st_size > 0 for name in names)
        weight = directory / "model.safetensors"
        return weight.is_file() and weight.stat().st_size > 0
    except (OSError, ValueError, KeyError):
        return False


def status(key: str = DEFAULT_MODEL, quantization: str = "4bit") -> dict:
    directory = model_path(key)
    missing = [name for name in ("transformers", "accelerate", "torch")
               if importlib.util.find_spec(name) is None]
    if quantization == "4bit" and importlib.util.find_spec("bitsandbytes") is None:
        missing.append("bitsandbytes")
    ready = weights_ready(key)
    reasons = (["缺少依赖：" + ", ".join(missing)] if missing else [])
    if not ready:
        reasons.append("本地模型未下载完整，请先下载模型")
    return {"model": key, "repo_id": MODELS[key], "path": str(directory),
            "weights_ready": ready, "available": ready and not missing,
            "missing_dependencies": missing,
            "loaded": _loaded == (key, quantization), "prompt_version": PROMPT_VERSION,
            "message": "；".join(reasons) if reasons else "模型已就绪，首次分析时加载",
            "models": [{"key": k, "name": v} for k, v in MODELS.items()]}


def download(key: str) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODELS[key], local_dir=str(model_path(key)),
                             allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja",
                                             "tokenizer.*", "merges.txt", "vocab.json"], max_workers=2)
    if not weights_ready(key):
        raise RuntimeError("下载结束但模型权重不完整，可重试继续下载")
    return str(path)


def unload() -> None:
    global _model, _processor, _loaded
    with _lock:
        _model, _processor, _loaded = None, None, None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load(key: str, quantization: str) -> tuple[Any, Any]:
    global _model, _processor, _loaded
    if _loaded == (key, quantization) and _model is not None:
        return _model, _processor
    state = status(key, quantization)
    if not state["available"]:
        raise RuntimeError(state["message"])
    unload()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("本地 VLM 需要 CUDA 显卡，请检查 PyTorch CUDA 环境")
    options: dict[str, Any] = {"local_files_only": True, "device_map": "auto",
                              "dtype": torch.float16, "attn_implementation": "sdpa"}
    if quantization == "4bit":
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(str(model_path(key)), **options)
    processor = AutoProcessor.from_pretrained(str(model_path(key)), local_files_only=True)
    _model, _processor, _loaded = model.eval(), processor, (key, quantization)
    return _model, _processor


def parse_response(text: str, allowed_states: list[str] | tuple[str, ...] | None = None) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as error:
        raise ValueError("VLM 未返回有效 JSON，请重试该目标") from error
    allowed_states = prompts.states(prompts.load()) if allowed_states is None else allowed_states
    if not isinstance(value, dict) or value.get("state") not in allowed_states:
        raise ValueError("VLM 返回了不支持的动作状态")
    if not isinstance(value.get("evidence"), str) or not value["evidence"].strip():
        raise ValueError("VLM 结果缺少可见证据")
    if not isinstance(value.get("uncertainty", ""), str):
        raise ValueError("VLM 不确定原因格式错误")
    return {"state": value["state"], "evidence": value["evidence"].strip()[:600],
            "uncertainty": value.get("uncertainty", "").strip()[:400]}


def cache_key(path: Path, key: str, quantization: str, prompt_config: dict | None = None) -> str:
    profile = prompts.load() if prompt_config is None else prompts.normalize(prompt_config)
    digest = hashlib.sha256(path.read_bytes())
    digest.update(f"{MODELS[key]}|{quantization}|{PROMPT_VERSION}|{prompts.identifier(profile)}|{prompts.compile_prompt(profile)}".encode())
    return digest.hexdigest()


def analyze(path: Path, key: str = DEFAULT_MODEL, quantization: str = "4bit", prompt_config: dict | None = None) -> dict:
    with _lock:
        import torch

        profile = prompts.load() if prompt_config is None else prompts.normalize(prompt_config)
        model, processor = _load(key, quantization)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        # Bound visual-token use without stretching or synthesizing detail.
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                  {"type": "text", "text": prompts.compile_prompt(profile)}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
        inputs = {name: value.to(model.device) for name, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        answer = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)[0]
        result = parse_response(answer, prompts.states(profile))
        return {**result, "model": MODELS[key], "quantization": quantization,
                "prompt_version": PROMPT_VERSION, "prompt_config_id": prompts.identifier(profile)}
