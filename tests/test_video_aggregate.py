"""Real OpenCV sampling of synthetic MP4, with simulated model outputs."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import app
import video_aggregate
from action_recognition import jobs, prompts, store, vlm
from web import common


class VideoAggregateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.videos = self.root / 'videos'
        self.videos.mkdir()
        writer = cv2.VideoWriter(str(self.videos / 'test.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(20):
            writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
        writer.release()
        self.detector = Mock(return_value={'detections': [{'bbox': [20, 10, 35, 25], 'confidence': .9}]})
        self.calls = []
        def analyze(path, model, quantization, profile, experiment):
            self.calls.append(path.name)
            state = '低头' if '抬头' in prompts.states(profile) else '写字'
            return {'state': state, 'evidence': '合成画面测试依据', 'uncertainty': ''}
        self.analyzer = Mock(side_effect=analyze)
        for module, name, value in [
            (store, 'ROOT', self.root / 'actions'), (common, 'VIDEO_DIR', self.videos),
            (common, '_require_head_detection', Mock()), (jobs, '_worker', threading.Lock()),
            (jobs, 'JOBS', {}), (jobs, '_events', {}),
            (vlm, 'status', Mock(return_value={'available': True})), (vlm, 'unload', Mock()),
            (vlm, 'analyze', self.analyzer), (jobs.vision, 'clear_model_cache', Mock()),
            (video_aggregate.head_fusion, 'detect_image', self.detector),
        ]:
            patched = patch.object(module, name, value)
            patched.start()
            self.addCleanup(patched.stop)
        self.client = TestClient(app.app)

    def wait(self, job_id):
        end = time.monotonic() + 5
        while time.monotonic() < end:
            response = self.client.get('/api/video-aggregate/jobs/' + job_id)
            job = response.json()
            if job['status'] in jobs.TERMINAL:
                with jobs._worker:
                    return job
            time.sleep(.01)
        self.fail('视频聚合任务未完成')

    def test_sample_interval_all_heads_then_context_and_persistent_results(self):
        response = self.client.post('/api/video-aggregate/jobs', json={'filename': 'test.mp4', 'interval_seconds': .5})
        self.assertEqual(response.status_code, 200, response.text)
        final = self.wait(response.json()['job_id'])
        self.assertEqual(final['status'], 'completed', final)
        run_id = final['run_id']
        result = self.client.get('/api/video-aggregate/runs/' + run_id).json()
        self.assertEqual([f['frame_index'] for f in result['video']['frames']], [0, 5, 10, 15])
        self.assertEqual([f['time_seconds'] for f in result['video']['frames']], [0, .5, 1, 1.5])
        self.assertEqual(self.detector.call_count, 4)
        self.assertEqual(self.analyzer.call_count, 8)
        self.assertTrue(all(name.endswith('_head.png') for name in self.calls[:4]))
        self.assertTrue(all(name.endswith('_target.png') for name in self.calls[4:]))
        self.assertTrue(all(f['counts'] == {'写字': 1} for f in result['video']['frames']))
        frame = result['video']['frames'][0]
        self.assertEqual(self.client.get(f'/api/video-aggregate/runs/{run_id}/files/{frame["result_file"]}').status_code, 200)
        detail = self.client.get(f'/api/video-aggregate/runs/{run_id}/frames/1').json()
        self.assertEqual(detail['targets'][0]['action']['state'], '写字')
        self.assertEqual(self.client.get(f'/api/video-aggregate/runs/{run_id}/export').status_code, 200)
        self.assertEqual(self.client.get(result['video_url']).status_code, 200)
        self.assertEqual(len(self.client.get('/api/video-aggregate/runs').json()['runs']), 1)

    def test_validation_bad_video_and_busy(self):
        for interval in (0, -.5, 86401):
            self.assertEqual(self.client.post('/api/video-aggregate/jobs', json={'filename': 'test.mp4', 'interval_seconds': interval}).status_code, 422)
        self.assertEqual(self.client.post('/api/video-aggregate/jobs', json={'filename': '../test.mp4'}).status_code, 400)
        self.assertEqual(self.client.get('/api/video-aggregate/runs/invalid').status_code, 400)
        (self.videos / 'invalid.mp4').write_bytes(b'invalid video')
        response = self.client.post('/api/video-aggregate/jobs', json={'filename': 'invalid.mp4'})
        final = self.wait(response.json()['job_id'])
        self.assertEqual(final['status'], 'failed')
        self.detector.assert_not_called()
        with jobs._worker:
            self.assertEqual(self.client.post('/api/video-aggregate/jobs', json={'filename': 'test.mp4'}).status_code, 409)

    def test_cancel_preserves_head_previews(self):
        def stop_after_head(*args):
            for event in jobs._events.values():
                event.set()
            return {'state': '低头', 'evidence': '合成头部姿态'}
        self.analyzer.side_effect = stop_after_head
        response = self.client.post('/api/video-aggregate/jobs', json={'filename': 'test.mp4', 'interval_seconds': 1})
        final = self.wait(response.json()['job_id'])
        self.assertEqual(final['status'], 'cancelled')
        result = self.client.get('/api/video-aggregate/runs/' + final['run_id']).json()
        self.assertEqual(len(result['video']['frames']), 2)
        self.assertEqual(result['video']['frames'][0]['counts'], {'低头': 1})
        self.assertTrue(result['video']['frames'][0]['provisional'])
        self.assertEqual(self.analyzer.call_count, 1)
