"""Serial, cancellable action jobs isolated from existing inference modules."""
import contextlib
import copy
import hashlib
import threading
import time
import uuid
from pathlib import Path

import head_fusion
import vision

from . import decoding, prompts, store, vlm
from .regions import Expansion

JOBS: dict[str, dict] = {}
_events: dict[str, threading.Event] = {}
_worker = threading.Lock()
_jobs_lock = threading.RLock()
TERMINAL = {"completed", "failed", "cancelled"}


@contextlib.contextmanager
def mutation():
    if not _worker.acquire(blocking=False):
        raise RuntimeError("动作识别任务正在运行，请等待完成或取消后再操作")
    try:
        yield
    finally:
        _worker.release()


def get(job_id: str) -> dict:
    with _jobs_lock:
        if job_id not in JOBS:
            raise FileNotFoundError("动作识别任务不存在，服务可能已重启，请重新打开批次")
        return copy.deepcopy(JOBS[job_id])


def active() -> dict | None:
    with _jobs_lock:
        return next((copy.deepcopy(job) for job in JOBS.values() if job["status"] not in TERMINAL), None)


def update(job_id: str, **values) -> None:
    with _jobs_lock:
        JOBS[job_id].update(values)


def cancel(job_id: str) -> dict:
    job = get(job_id)
    if job["kind"] == "download":
        raise ValueError("模型下载正在写入权重文件，请等待下载完成；失败后可继续下载")
    if job["status"] not in TERMINAL:
        _events[job_id].set()
        update(job_id, message="已请求取消，当前目标处理结束后停止")
    return get(job_id)


def _launch(kind: str, run_id: str | None, total: int, work) -> dict:
    # Caller owns _worker until the worker's finally block releases it.
    job_id = uuid.uuid4().hex
    event = threading.Event()
    job = {"job_id": job_id, "kind": kind, "run_id": run_id, "status": "queued",
           "created_at": store.now(), "total": total, "progress": 0, "errors": 0,
           "cached": 0, "message": "任务已排队"}
    with _jobs_lock:
        JOBS[job_id], _events[job_id] = job, event

    def execute():
        update(job_id, status="running")
        try:
            work(job_id, event)
            update(job_id, status="cancelled" if event.is_set() else "completed",
                   message="已取消，已完成的结果已保留" if event.is_set() else "处理完成")
        except Exception as error:
            update(job_id, status="failed", message=str(error))
        finally:
            update(job_id, finished_at=store.now())
            _worker.release()

    threading.Thread(target=execute, name=f"action-{kind}", daemon=True).start()
    return copy.deepcopy(job)


def start_prepare(paths: list[Path], expansion: Expansion) -> dict:
    if not _worker.acquire(blocking=False):
        raise RuntimeError("已有动作识别任务正在运行")
    try:
        record = store.create([path.name for path in paths], expansion)
        run_id = record["run_id"]

        def work(job_id, event):
            try:
                vlm.unload()
                for index, source in enumerate(paths, 1):
                    if event.is_set():
                        break
                    update(job_id, message=f"Head 检测与区域裁剪：{source.name}")
                    try:
                        source_path, image = store.snapshot(record, source, index)
                        prediction = head_fusion.detect_image(source_path)
                        store.append_image(record, source.name, index, image,
                                           prediction["detections"], expansion)
                        record["head_parameters"] = head_fusion.parameters()
                    except Exception as error:
                        record["errors"].append({"source": source.name, "message": str(error)})
                    store.write(record)
                    update(job_id, progress=index, errors=len(record["errors"]))
                record["status"] = ("cancelled" if event.is_set() else
                                    "failed" if not record["images"] and record["errors"] else "ready")
            except Exception:
                record["status"] = "failed"
                raise
            finally:
                store.write(record)
            if not record["images"] and record["errors"] and not event.is_set():
                raise RuntimeError(record["errors"][0]["message"])

        return _launch("prepare", run_id, len(paths), work)
    except Exception:
        _worker.release()
        raise


def start_analysis(run_id: str, crop_ids: list[str], model: str, quantization: str, force: bool, experiment: str = 'A') -> dict:
    if not _worker.acquire(blocking=False):
        raise RuntimeError("已有动作识别任务正在运行")
    try:
        record = store.read(run_id)
        ids = list(dict.fromkeys(crop_ids)) if crop_ids else [item["crop_id"] for item in record["crops"]]
        if not ids:
            raise ValueError("该批次没有可分析的裁剪，请先检测并生成区域")
        items = [store.find_crop(record, crop_id) for crop_id in ids]
        # A running batch keeps one immutable profile even if the user saves new rules.
        profile = prompts.load()
        profile_id = prompts.identifier(profile)
        decode = decoding.describe(experiment)
        ready = vlm.status(model, quantization)
        if not ready["available"]:
            raise ValueError(ready["message"])

        def work(job_id, event):
            record.setdefault("prompt_profiles", {})[profile_id] = {
                "config": profile, "compiled_prompt": prompts.compile_prompt(profile),
                "prompt_version": prompts.VERSION}
            record["last_prompt_config_id"] = profile_id
            store.write(record)
            session = {'session_id': job_id, 'started_at': store.now(), 'status': 'running',
                       'decoding': decode, 'model': vlm.MODELS[model], 'quantization': quantization,
                       'prompt_config_id': profile_id, 'force': force, 'total': len(items), 'results': []}
            record.setdefault('analysis_sessions', []).append(session)
            store.write(record)
            session_started = time.perf_counter()
            try:
                _analyze_items(job_id, event, record, items, model, quantization, force, profile, profile_id, experiment, session)
                session['status'] = 'cancelled' if event.is_set() else 'completed'
            except Exception:
                session['status'] = 'failed'
                raise
            finally:
                session['finished_at'] = store.now()
                session['elapsed_seconds'] = round(time.perf_counter() - session_started, 3)
                store.write(record)

        return _launch("analyze", run_id, len(items), work)
    except Exception:
        _worker.release()
        raise


def _analyze_items(job_id, event, record, items, model, quantization, force, profile, profile_id, experiment, session):
    # Free Head detector caches before loading the VLM on a 12 GB GPU.
    vision.clear_model_cache()
    for index, item in enumerate(items, 1):
        if event.is_set():
            break
        path = store.folder(record['run_id']) / item["marked_file"]
        update(job_id, message=f"动作分析 {index}/{len(items)}：{item['crop_id']}（首次需要加载模型）")
        started = time.perf_counter()
        measurement = {'crop_id': item['crop_id'], 'source': item['source'],
                       'region_bbox': list(item['region_bbox']), 'revision': item['revision']}
        try:
            measurement['input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            key = vlm.cache_key(path, model, quantization, profile, experiment)
            if not force and item.get("analysis_key") == key and item.get("status") == "done":
                update(job_id, cached=get(job_id)["cached"]+1)
                measurement['cached'] = True
            else:
                result = vlm.analyze(path, model, quantization, profile, experiment)
                item.update({"action": {**result, "decoding": decoding.describe(experiment), "analyzed_at": store.now(), "prompt_config_id": profile_id}, "status": "done",
                             "analysis_key": key, "elapsed_seconds": round(time.perf_counter()-started, 2)})
                item.pop("error", None)
            measurement.update(status='done', action=copy.deepcopy(item['action']))
        except Exception as error:
            item.update({"status": "failed", "error": str(error), "action": None, "analysis_key": None})
            item['elapsed_seconds'] = round(time.perf_counter()-started, 2)
            update(job_id, errors=get(job_id)["errors"]+1)
            measurement.update(status='failed', error=str(error),
                               metrics=getattr(error, 'metrics', None),
                               raw_response=getattr(error, 'raw_response', None))
            # A failed model load / CUDA OOM is shared by every target; don't repeat it hundreds of times.
            if isinstance(error, RuntimeError):
                raise
        finally:
            measurement['elapsed_seconds'] = round(time.perf_counter()-started, 3)
            session['results'].append(measurement)
            store.write(record)
        update(job_id, progress=index)


def start_download(model: str) -> dict:
    vlm.model_path(model)
    if not _worker.acquire(blocking=False):
        raise RuntimeError("已有动作识别任务正在运行")

    def work(job_id, event):
        update(job_id, message=f"从 Qwen 官方仓库下载 {vlm.MODELS[model]}，可重试续传")
        vlm.download(model)
        update(job_id, progress=1)

    return _launch("download", None, 1, work)
