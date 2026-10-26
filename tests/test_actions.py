"""Independent action workflow with synthetic pictures and deterministic VLM responses."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

import app
from action_recognition import jobs, prompts, store, vlm
from action_recognition.regions import Expansion, estimate, render_crop
from web import common


class RegionTests(unittest.TestCase):
    def test_expansion_uses_head_width_and_height_and_clips_to_image(self):
        self.assertEqual(estimate([40, 10, 60, 30], 160, 120, Expansion()), [23, 10, 77, 56])
        self.assertEqual(estimate([130, 90, 150, 110], 160, 120, Expansion()), [113, 90, 160, 120])

    def test_rectangular_crop_keeps_scene_around_target_without_polygon_mask(self):
        source = Image.new('RGB', (160, 120), 'white')
        head, region = [40, 10, 60, 30], [27, 8, 73, 60]
        raw, masked = render_crop(source, region, head)
        self.assertEqual(raw.size, (46, 52))
        self.assertEqual(raw.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(masked.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(masked.getpixel((23, 12)), (255, 255, 255))
        self.assertEqual(masked.getpixel((23, 36)), (255, 255, 255))

    def test_invalid_and_nonfinite_coordinates_are_rejected(self):
        for box in ([0, 0, 0, 4], [200, 0, 220, 3], [1, float('nan'), 3, 4], [1, 2, 3]):
            with self.assertRaises(ValueError):
                estimate(box, 160, 120, Expansion())
        with self.assertRaises(ValueError):
            Expansion(below=float('inf'))


class VlmResponseTests(unittest.TestCase):
    def test_requested_five_actions_and_unknown_are_the_only_current_labels(self):
        self.assertEqual(set(vlm.STATES), {'抬头', '低头', '低头玩手机', '低头看书', '睡觉', '无法判断'})
        for state in vlm.STATES:
            self.assertEqual(vlm.parse_response(json.dumps({'state': state, 'evidence': '可见姿态证据'}), vlm.STATES)['state'], state)
        with self.assertRaises(ValueError):
            vlm.parse_response(json.dumps({'state': '疑似阅读', 'evidence': '旧类别'}), vlm.STATES)

    def test_single_state_and_evidence_are_required(self):
        value = vlm.parse_response(json.dumps({'state': '低头', 'evidence': '头部向下', 'uncertainty': '手被遮挡'}), vlm.STATES)
        self.assertEqual(value['state'], '低头')
        for text in ('not JSON', '{}', '{"state":"专心","evidence":"x"}', '{"state":"低头","evidence":[]}'):
            with self.assertRaises(ValueError):
                vlm.parse_response(text, vlm.STATES)

    def test_unknown_is_a_valid_result_and_fences_are_tolerated(self):
        text = json.dumps({'state': '无法判断', 'evidence': '目标被遮挡'})
        self.assertEqual(vlm.parse_response('```json\n'+text+'\n```', vlm.STATES)['state'], '无法判断')


class ActionApiTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.uploads = self.root / 'uploads'; self.uploads.mkdir()
        Image.new('RGB', (160, 120), '#c5d5e5').save(self.uploads / 'class.png')
        self.detector = Mock(return_value={'detections': [
            {'bbox': [40, 10, 60, 30], 'confidence': .9},
            {'bbox': [130, 90, 150, 110], 'confidence': .8}]})
        self.analyzer = Mock(return_value={'state': '低头', 'evidence': '目标头部向下', 'uncertainty': '',
                                          'model': vlm.MODELS['4b'], 'prompt_version': vlm.PROMPT_VERSION})
        for module, name, value in (
            (store, 'ROOT', self.root / 'actions'), (common, 'UPLOAD_DIR', self.uploads),
            (common, '_require_head_detection', Mock()), (jobs, '_worker', threading.Lock()),
            (jobs, 'JOBS', {}), (jobs, '_events', {}),
            (jobs.head_fusion, 'detect_image', self.detector),
            (jobs.vision, 'clear_model_cache', Mock()), (vlm, 'unload', Mock()),
            (vlm, 'status', Mock(return_value={'available': True, 'message': 'ready'})),
            (vlm, 'analyze', self.analyzer),
        ):
            mocked = patch.object(module, name, value); mocked.start(); self.addCleanup(mocked.stop)
        self.client = TestClient(app.app)

    def wait(self, job_id):
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            job = jobs.get(job_id)
            if job['status'] in jobs.TERMINAL:
                # Status can be visible just before execute() releases the worker lock.
                with jobs._worker:
                    return jobs.get(job_id)
            time.sleep(.01)
        self.fail('后台动作任务未及时完成')

    def prepare(self):
        response = self.client.post('/api/actions/runs', json={'filenames': ['class.png']})
        self.assertEqual(response.status_code, 200, response.text)
        job = self.wait(response.json()['job_id'])
        self.assertEqual(job['status'], 'completed', job)
        return self.client.get('/api/actions/runs/'+job['run_id']).json()

    def analyze(self, run_id, **parameters):
        response = self.client.post(f'/api/actions/runs/{run_id}/analyze', json=parameters)
        self.assertEqual(response.status_code, 200, response.text)
        return self.wait(response.json()['job_id'])

    def test_prepare_snapshots_and_crops_persist_without_pose(self):
        record = self.prepare()
        self.assertEqual(record['crop_count'], 2)
        self.assertEqual(record['crops'][0]['crop_id'], 'HC001-001')
        self.assertEqual(record['crops'][0]['region_bbox'], [23, 10, 77, 56])
        self.assertEqual(record['crops'][0]['roi_shape'], 'head_context_rectangle_v1')
        self.assertEqual(len(record['crops'][0]['roi_polygon']), 4)
        self.assertEqual(self.detector.call_count, 1)
        self.analyzer.assert_not_called()
        self.assertEqual(record['head_parameters']['base']['weights'], 'medium.pt')
        directory = store.folder(record['run_id'])
        self.assertTrue((directory / 'sources/001.png').is_file())
        crop = record['crops'][0]
        raw = self.client.get(f"/api/actions/runs/{record['run_id']}/files/{crop['raw_file']}")
        self.assertEqual(raw.status_code, 200)
        self.assertIn('no-store', raw.headers['cache-control'])
        self.assertEqual(self.client.get('/api/actions/runs').json()['runs'][0]['crop_count'], 2)

    def test_selected_analysis_cache_and_manual_edit_invalidation(self):
        record = self.prepare(); run_id = record['run_id']; crop_id = record['crops'][0]['crop_id']
        self.analyze(run_id, crop_ids=[crop_id])
        self.assertEqual(self.analyzer.call_count, 1)
        saved = store.read(run_id)
        self.assertEqual(saved['crops'][0]['action']['state'], '低头')
        self.assertIsNone(saved['crops'][1]['action'])
        job = self.analyze(run_id, crop_ids=[crop_id])
        self.assertEqual(job['cached'], 1)
        self.assertEqual(self.analyzer.call_count, 1)
        self.analyze(run_id, crop_ids=[crop_id], force=True)
        self.assertEqual(self.analyzer.call_count, 2)
        response = self.client.patch(f'/api/actions/runs/{run_id}/crops/{crop_id}', json={'bbox': [10, 0, 90, 115]})
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()['crops'][0]
        self.assertEqual(item['status'], 'pending'); self.assertIsNone(item['action'])
        self.assertNotIn('elapsed_seconds', item)
        self.analyze(run_id, crop_ids=[crop_id])
        self.assertEqual(self.analyzer.call_count, 3)

    def test_expansion_updates_all_crops_and_clears_results(self):
        record = self.prepare(); run_id = record['run_id']; self.analyze(run_id)
        response = self.client.post(f'/api/actions/runs/{run_id}/expansion', json={'side': 0, 'top': 0, 'below': 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['crops'][0]['region_bbox'], [40, 10, 60, 50])
        self.assertTrue(all(item['action'] is None for item in response.json()['crops']))

    def test_region_must_include_its_head_and_paths_are_confined(self):
        record = self.prepare(); run_id = record['run_id']
        for box in ([0, 0, 10, 10], [0, 0, 0, 10], ['NaN', 0, 50, 50]):
            response = self.client.patch(f'/api/actions/runs/{run_id}/crops/HC001-001', json={'bbox': box})
            self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.client.get('/api/actions/runs/bad').status_code, 400)
        self.assertEqual(self.client.get(f'/api/actions/runs/{run_id}/files/record.json').status_code, 404)
        with self.assertRaises(FileNotFoundError):
            store.image_file(run_id, '../outside.png')
        self.assertEqual(self.client.post('/api/actions/runs', json={'filenames': ['../class.png']}).status_code, 400)

    def test_busy_guard_and_cancel_keep_completed_results(self):
        record = self.prepare(); run_id = record['run_id']
        def analyze(*args):
            active = jobs.active()
            assert active is not None
            busy_response = self.client.patch(f'/api/actions/runs/{run_id}/crops/HC001-001', json={'bbox': [0, 0, 100, 120]})
            self.assertEqual(busy_response.status_code, 409)
            self.assertEqual(self.client.post('/api/actions/model/unload').status_code, 409)
            jobs.cancel(active['job_id'])
            return {'state': '无法判断', 'evidence': '头部遮挡', 'uncertainty': '看不清'}
        self.analyzer.side_effect = analyze
        job = self.analyze(run_id)
        self.assertEqual(job['status'], 'cancelled')
        saved = store.read(run_id)
        self.assertEqual(saved['crops'][0]['action']['state'], '无法判断')
        self.assertIsNone(saved['crops'][1]['action'])

    def test_model_failure_is_reported_without_a_fabricated_action(self):
        record = self.prepare(); run_id = record['run_id']
        self.analyzer.side_effect = RuntimeError('CUDA out of memory')
        job = self.analyze(run_id)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(self.analyzer.call_count, 1)
        self.assertEqual(store.read(run_id)['crops'][0]['status'], 'failed')
        self.assertIsNone(store.read(run_id)['crops'][0]['action'])

    def test_invalid_model_or_missing_weights_does_not_start_inference(self):
        record = self.prepare(); run_id = record['run_id']
        self.assertEqual(self.client.post(f'/api/actions/runs/{run_id}/analyze', json={'model': 'other'}).status_code, 422)
        with patch.object(vlm, 'status', return_value={'available': False, 'message': '本地模型未下载完整'}):
            response = self.client.post(f'/api/actions/runs/{run_id}/analyze', json={})
        self.assertEqual(response.status_code, 400)
        self.analyzer.assert_not_called()

    def test_custom_action_profile_persists_and_preview_does_not_save(self):
        original = self.client.get('/api/actions/prompt').json()
        config = original['config']
        yawn = next(item for item in config['actions'] if item['name'] == '打哈欠')
        self.assertFalse(yawn['enabled'])
        yawn['enabled'] = True
        preview = self.client.post('/api/actions/prompt/preview', json=config)
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertIn('打哈欠', preview.json()['states'])
        self.assertNotIn('打哈欠', self.client.get('/api/actions/prompt').json()['states'])
        saved = self.client.put('/api/actions/prompt', json=config)
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(self.client.get('/api/actions/prompt').json()['config_id'], saved.json()['config_id'])
        self.assertTrue((store.ROOT / 'prompt_config.json').is_file())
        self.assertIn('打哈欠', saved.json()['compiled_prompt'])
        result = vlm.parse_response(json.dumps({'state': '打哈欠', 'evidence': '目标嘴部明显张大'}), saved.json()['states'])
        self.assertEqual(result['state'], '打哈欠')

    def test_profile_changes_invalidate_cache_without_erasing_old_results(self):
        record = self.prepare(); run_id = record['run_id']; crop_id = record['crops'][0]['crop_id']
        self.analyze(run_id, crop_ids=[crop_id]); old = store.read(run_id)['crops'][0]['action']['prompt_config_id']
        config = self.client.get('/api/actions/prompt').json()['config']
        config['instructions'] += '新规则说明'
        self.assertEqual(self.client.put('/api/actions/prompt', json=config).status_code, 200)
        self.assertEqual(store.read(run_id)['crops'][0]['action']['prompt_config_id'], old)
        self.analyze(run_id, crop_ids=[crop_id])
        saved = store.read(run_id)
        self.assertEqual(self.analyzer.call_count, 2)
        self.assertNotEqual(saved['crops'][0]['action']['prompt_config_id'], old)
        self.assertEqual(len(saved['prompt_profiles']), 2)

    def test_inflight_batch_uses_one_snapshot_when_configuration_is_saved(self):
        record = self.prepare(); run_id = record['run_id']
        before = prompts.load(); seen = []
        def analyze(path, model, quantization, profile):
            seen.append(prompts.identifier(profile))
            changed = prompts.defaults(); changed['instructions'] = '保存到下一次任务'
            prompts.save(changed)
            return {'state': '低头', 'evidence': '目标头部向下', 'uncertainty': ''}
        self.analyzer.side_effect = analyze
        self.analyze(run_id)
        self.assertEqual(seen, [prompts.identifier(before)]*2)
        self.assertNotEqual(prompts.identifier(prompts.load()), seen[0])

    def test_invalid_profiles_are_rejected(self):
        config = prompts.defaults()
        config['actions'][1]['name'] = config['actions'][0]['name']
        self.assertEqual(self.client.put('/api/actions/prompt', json=config).status_code, 422)
        config = prompts.defaults()
        for action in config['actions']:
            action['enabled'] = False
        self.assertEqual(self.client.put('/api/actions/prompt', json=config).status_code, 422)
        config = prompts.defaults(); config['actions'][0]['rule'] = '  '
        self.assertEqual(self.client.put('/api/actions/prompt', json=config).status_code, 422)


if __name__ == '__main__':
    unittest.main()
