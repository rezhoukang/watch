"""Persistent batch-wide Head pass, then expanded-context pass."""
import copy
import hashlib
import time

from . import decoding, head_only, prompts, store, two_stage


def run(job_id, event, record, items, model, quantization, force, profile, experiment, session):
    from . import jobs

    entries = []
    context = session['context_profile']
    root = store.folder(record['run_id'])
    for item in items:
        if event.is_set():
            return
        head_path, head_file = head_only.input_file(record, item)
        context_path = root / item['marked_file']
        key = two_stage.cache_key(head_path, context_path, model, quantization, profile, context, experiment)
        measurement = {'crop_id': item['crop_id'], 'source': item['source'], 'revision': item['revision'],
                       'region_bbox': list(item['head_bbox']), 'context_region_bbox': list(item['region_bbox']),
                       'input_sha256': hashlib.sha256(head_path.read_bytes()).hexdigest(),
                       'context_input_sha256': hashlib.sha256(context_path.read_bytes()).hexdigest(),
                       'head_only': True, 'two_stage': True}
        session['results'].append(measurement)
        if not force and item.get('analysis_key') == key and item.get('status') == 'done':
            measurement.update(status='done', cached=True, action=copy.deepcopy(item['action']))
            jobs.update(job_id, cached=jobs.get(job_id)['cached'] + 1)
            continue
        item.update(status='pending', action=None, analysis_key=None)
        item.pop('error', None)
        item.pop('elapsed_seconds', None)
        entries.append({'item': item, 'measurement': measurement, 'key': key, 'started': None,
                        'head_path': head_path, 'head_file': head_file, 'head_bbox': list(item['head_bbox']),
                        'context_path': context_path, 'context_file': item['marked_file'],
                        'context_bbox': list(item['region_bbox'])})
    store.write(record)

    def progress(phase, current, total):
        label = '第一阶段：全部 Head 抬头/低头' if phase == 'head' else '第二阶段：全部扩展 ROI 补充'
        jobs.update(job_id, stage=phase, stage_current=current, stage_total=total,
                    progress=jobs.get(job_id)['cached'] + (current / 2 if phase == 'head' else (total + current) / 2),
                    message=f'{label} {current}/{total}')

    def publish(entry, phase, result, error):
        item, measurement = entry['item'], entry['measurement']
        if error:
            item.update(status='failed', error=str(error), analysis_key=None)
            measurement.pop('action', None)
            measurement.update(status='failed', error=str(error),
                               stages={'head': result} if result else {},
                               metrics=getattr(error, 'metrics', None), raw_response=getattr(error, 'raw_response', None))
            jobs.update(job_id, errors=jobs.get(job_id)['errors'] + 1)
        else:
            action = {**result, 'head_only': True, 'two_stage': True,
                      'provisional': phase == 'head', 'decoding': decoding.describe(experiment),
                      'analyzed_at': store.now(), 'prompt_config_id': prompts.identifier(profile),
                      'input_file': entry[f'{phase}_file']}
            item.update(action=action, status='pending_context' if phase == 'head' else 'done',
                        analysis_key=None if phase == 'head' else entry['key'])
            measurement.update(status=item['status'], action=copy.deepcopy(action))
            if phase == 'head':
                measurement['stages'] = {'head': copy.deepcopy(result)}
            item.pop('error', None)
        store.write(record)

    # Track compute time per target, excluding time spent on other targets between passes.
    started = {}
    def timed_progress(phase, current, total):
        if (phase, current - 1) in started:
            entry = entries[current - 1]
            elapsed = time.perf_counter() - started.pop((phase, current - 1))
            entry['measurement']['elapsed_seconds'] = round(entry['measurement'].get('elapsed_seconds', 0) + elapsed, 3)
            entry['item']['elapsed_seconds'] = entry['measurement']['elapsed_seconds']
        if current < total:
            started[(phase, current)] = time.perf_counter()
        progress(phase, current, total)

    two_stage.run_batch(entries, model, quantization, profile, context, experiment,
                        timed_progress, publish, event.is_set)
    for row in session['results']:
        if event.is_set() and row.get('status') not in ('done', 'failed'):
            row['status'] = 'cancelled'
    store.write(record)
