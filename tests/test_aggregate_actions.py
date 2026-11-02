"""Synthetic-image checks for aggregate VLM routing; no real model inference."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

import aggregate
import aggregate_actions
import aggregate_jobs
import app
from action_recognition import jobs, prompts
from web import common


class AggregateVlmTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        uploads = self.root / 'uploads'
        uploads.mkdir()
        Image.new('RGB', (120, 100), 'white').save(uploads / 'class.png')
        self.analyzer = Mock(return_value={'state': '低头', 'evidence': '目标头部朝下', 'uncertainty': ''})
        self.detector = Mock(side_effect=lambda _: {
            'detections': [{'class_name': 'head', 'confidence': .9, 'bbox': box}
                           for box in [[20, 10, 40, 30], [70, 20, 90, 40]]],
            'mode': 'head', 'weight': 'medium.pt', 'imgsz': 640,
            'classes_used': [], 'fusion': {}, 'rule': 'head rule'})
        self.pose = Mock(return_value={'detections': [], 'excluded_unknown': 0})
        for module, name, value in [
            (common, 'UPLOAD_DIR', uploads), (common, '_require_head_detection', Mock()),
            (aggregate, 'RESULT_DIR', self.root / 'results'), (aggregate, 'detect_targets', self.detector),
            (aggregate, 'detect_poses', self.pose),
            (aggregate_actions.vlm, 'status', Mock(return_value={'available': True})),
            (aggregate_actions.vlm, 'unload', Mock()), (aggregate_actions.vlm, 'analyze', self.analyzer),
            (aggregate_actions.vision, 'clear_model_cache', Mock()), (jobs, '_worker', threading.Lock()),
            (aggregate_jobs, 'JOBS', {}), (aggregate_jobs, '_runner', threading.Lock()),
        ]:
            patched = patch.object(module, name, value)
            patched.start(); self.addCleanup(patched.stop)
        self.client = TestClient(app.app)

    def post(self, **options):
        return self.client.post('/api/aggregate/image/class.png', json={'face': False, **options})

    def wait_job(self, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            value = self.client.get('/api/aggregate/jobs/' + job_id).json()
            if value['status'] in aggregate_jobs.TERMINAL:
                with aggregate_jobs._runner:
                    return value
            time.sleep(.01)
        self.fail('聚合后台任务未及时结束')

    def test_background_job_exposes_live_target_progress_and_survives_status_queries(self):
        release = threading.Event(); started = threading.Event()
        self.addCleanup(release.set)
        def analyze(*args):
            started.set()
            release.wait(4)
            return {'state': '低头', 'evidence': '头部朝下'}
        self.analyzer.side_effect = analyze
        response = self.client.post('/api/aggregate/jobs/class.png', json={'face': False, 'pose': True, 'multimodal': True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(started.wait(3))
        job_id = response.json()['job_id']
        live = self.client.get('/api/aggregate/jobs/' + job_id).json()
        self.assertEqual((live['stage'], live['current'], live['total']), ('multimodal', 0, 2))
        self.assertEqual(self.client.get('/api/aggregate/jobs/active').json()['job']['job_id'], job_id)
        self.assertEqual(self.client.post('/api/aggregate/jobs/class.png', json={}).status_code, 409)
        release.set()
        final = self.wait_job(job_id)
        self.assertEqual((final['status'], final['percent']), ('completed', 100))
        self.assertTrue(all(s['status'] == 'completed' for s in final['stages']))
        self.assertEqual(final['result']['summary']['vlm_actions'], 2)

    def test_background_failure_reports_actual_stage_without_claiming_completion(self):
        self.detector.side_effect = ValueError('目标检测失败')
        response = self.client.post('/api/aggregate/jobs/class.png', json={'face': False, 'pose': False})
        job = self.wait_job(response.json()['job_id'])
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['stage'], 'detect')
        self.assertLess(job['percent'], 100)
        self.assertEqual(job['error'], '目标检测失败')
        self.assertEqual(self.client.get('/api/aggregate/jobs/missing').status_code, 404)

    def test_progress_events_include_each_processed_target_and_skipped_steps_are_absent(self):
        events = []
        src = self.root / 'uploads' / 'class.png'
        aggregate.aggregate_image(src, face=False, pose=True, multimodal=True, save=False, progress=events.append)
        steps = list(dict.fromkeys(event['stage'] for event in events))
        self.assertEqual(steps, ['prepare', 'detect', 'multimodal', 'render'])
        done = [event['current'] for event in events if event['stage'] == 'multimodal']
        self.assertIn(1, done); self.assertEqual(done[-1], 2)

    def test_both_switches_use_head_crops_and_saved_prompt_instead_of_pose(self):
        before = prompts.load()
        response = self.post(pose=True, multimodal=True)
        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()
        self.pose.assert_not_called()
        self.assertEqual(self.analyzer.call_count, 2)
        self.assertEqual(record['summary']['vlm_actions'], 2)
        self.assertEqual(record['poses'], [])
        self.assertEqual(record['targets'][0]['action']['region_bbox'], [3, 10, 57, 56])
        self.assertEqual(record['targets'][1]['action']['state'], '低头')
        self.assertEqual(self.analyzer.call_args.args[3], before)
        self.assertEqual(record['steps']['multimodal']['experiment'], 'A')
        self.assertTrue(list((self.root / 'results' / 'vlm').glob('*/record.json')))
        self.assertTrue(list((self.root / 'results' / 'vlm').glob('*/HC0001_target.png')))

    def test_without_multimodal_original_pose_path_is_used(self):
        self.assertEqual(self.post(pose=True, multimodal=False).status_code, 200)
        self.pose.assert_called_once()
        self.analyzer.assert_not_called()

    def test_multimodal_without_pose_does_not_run_either_action_path(self):
        response = self.post(pose=False, multimodal=True)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()['steps']['multimodal']['enabled'])
        self.pose.assert_not_called(); self.analyzer.assert_not_called()

    def test_failure_is_explicit_and_other_target_can_complete(self):
        self.analyzer.side_effect = [ValueError('缺少可见证据'), {'state': '无法判断', 'evidence': '头部被遮挡'}]
        response = self.post(pose=True, multimodal=True)
        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()
        self.assertEqual(record['summary']['failed_actions'], 1)
        self.assertEqual(record['targets'][0]['action']['status'], 'failed')
        self.assertEqual(record['targets'][1]['action']['state'], '无法判断')
        self.pose.assert_not_called()

    def test_model_unavailable_and_busy_fail_before_head_detection(self):
        with patch.object(aggregate_actions.vlm, 'status', return_value={'available': False, 'message': '模型未就绪'}):
            self.assertEqual(self.post(pose=True, multimodal=True).status_code, 400)
        with jobs._worker:
            self.assertEqual(self.post(pose=True, multimodal=True).status_code, 409)
        self.detector.assert_not_called()


if __name__ == '__main__':
    unittest.main()
