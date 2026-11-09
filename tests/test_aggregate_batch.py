"""Batch queue, failure isolation and stable input/result snapshots."""
import tempfile
import io
import json
import zipfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

import aggregate
import aggregate_batch
import aggregate_jobs
import app
from web import common


class AggregateBatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.uploads = self.root / 'uploads'
        self.uploads.mkdir()
        for name in ('a.png', 'b.png', 'c.png'):
            Image.new('RGB', (32, 24), 'white').save(self.uploads / name)
        self.calls = []
        def analyze(source, **options):
            self.calls.append((source.name, options))
            options['progress']({'stage': 'detect', 'message': '模拟目标检测', 'current': 1, 'total': 1})
            aggregate.RESULT_DIR.mkdir(parents=True, exist_ok=True)
            result = source.stem + '_agg.png'
            Image.new('RGB', (32, 24), 'blue').save(aggregate.RESULT_DIR / result)
            return {'result': result, 'targets': [], 'count': 0, 'summary': {}, 'elapsed_seconds': .01, 'image_width': 32, 'image_height': 24}
        self.analyzer = Mock(side_effect=analyze)
        for module, name, value in [
            (common, 'UPLOAD_DIR', self.uploads), (common, '_require_head_detection', Mock()),
            (aggregate, 'RESULT_DIR', self.root / 'results'), (aggregate, 'aggregate_image', self.analyzer),
            (aggregate_jobs, 'JOBS', {}), (aggregate_jobs, '_runner', threading.Lock()),
        ]:
            patched = patch.object(module, name, value)
            patched.start(); self.addCleanup(patched.stop)
        self.client = TestClient(app.app)

    def wait(self, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = self.client.get('/api/aggregate/jobs/' + job_id).json()
            if job['status'] in aggregate_jobs.TERMINAL:
                with aggregate_jobs._runner:
                    return job
            time.sleep(.01)
        self.fail('批量任务没有完成')

    def test_deduplicated_serial_batch_persists_images_and_results(self):
        response = self.client.post('/api/aggregate/batch', json={'filenames': ['a.png', 'b.png', 'a.png'], 'multimodal': True})
        self.assertEqual(response.status_code, 200, response.text)
        result = self.wait(response.json()['job_id'])
        self.assertEqual((result['status'], result['completed_images'], result['failed_images']), ('completed', 2, 0))
        self.assertEqual([row[0] for row in self.calls], ['a.png', 'b.png'])
        self.assertTrue(all(not row[1]['face'] and not row[1]['pose'] and row[1]['multimodal'] for row in self.calls))
        self.assertTrue((aggregate_batch.directory(result['job_id']) / 'record.json').is_file())
        row = result['results'][0]['result']
        (self.uploads / 'a.png').unlink()
        self.assertEqual(self.client.get(row['source_url']).status_code, 200)
        self.assertEqual(self.client.get(row['result_url']).status_code, 200)
        history = self.client.get('/api/aggregate/history').json()['records']
        exported = self.client.post('/api/aggregate/history/export', json={'record_ids': [item['id'] for item in history]})
        self.assertEqual(exported.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            self.assertEqual(set(archive.namelist()), {'results.json', 'viewer.html'})
            payload = json.loads(archive.read('results.json'))
            self.assertEqual({row['source'] for row in payload['records']}, {'a.png', 'b.png'})
            self.assertTrue(all(row['result']['source_url'].startswith('data:image/png;base64,') for row in payload['records']))
            self.assertIn('显示为点', archive.read('viewer.html').decode('utf-8'))
        aggregate_jobs.JOBS.clear()
        restored = self.client.get('/api/aggregate/jobs/' + result['job_id']).json()
        self.assertEqual(restored['completed_images'], 2)

    def test_one_error_keeps_processing_other_images(self):
        normal = self.analyzer.side_effect
        def analyze(source, **options):
            if source.name == 'b.png':
                raise ValueError('图片损坏')
            return normal(source, **options)
        self.analyzer.side_effect = analyze
        response = self.client.post('/api/aggregate/batch', json={'filenames': ['a.png', 'b.png', 'c.png']})
        job = self.wait(response.json()['job_id'])
        self.assertEqual((job['completed_images'], job['failed_images']), (2, 1))
        self.assertEqual(job['results'][1]['error'], '图片损坏')
        self.assertEqual(job['results'][2]['status'], 'completed')

    def test_validation_and_shared_runner(self):
        self.assertEqual(self.client.post('/api/aggregate/batch', json={'filenames': []}).status_code, 422)
        self.assertEqual(self.client.post('/api/aggregate/batch', json={'filenames': ['a.png', 'missing.png']}).status_code, 404)
        self.assertEqual(self.client.post('/api/aggregate/batch', json={'filenames': ['../a.png']}).status_code, 400)
        self.assertEqual(self.client.post('/api/aggregate/batch', json={'filenames': ['a.png'], 'face': True}).status_code, 422)
        self.analyzer.assert_not_called()
        with aggregate_jobs._runner:
            self.assertEqual(self.client.post('/api/aggregate/batch', json={'filenames': ['a.png']}).status_code, 409)

    def test_stop_keeps_completed_result_and_skips_remaining_images(self):
        entered, release = threading.Event(), threading.Event()
        normal = self.analyzer.side_effect

        def analyze(source, **options):
            if source.name == 'b.png':
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('测试等待超时')
            return normal(source, **options)

        self.analyzer.side_effect = analyze
        response = self.client.post('/api/aggregate/batch', json={'filenames': ['a.png', 'b.png', 'c.png']})
        job_id = response.json()['job_id']
        try:
            self.assertTrue(entered.wait(5))
            stopped = self.client.post(f'/api/aggregate/jobs/{job_id}/cancel')
            self.assertEqual(stopped.status_code, 200)
            self.assertTrue(stopped.json()['cancel_requested'])
        finally:
            release.set()
        job = self.wait(job_id)
        self.assertEqual(job['status'], 'cancelled')
        self.assertEqual((job['completed_images'], job['failed_images']), (1, 0))
        self.assertEqual([row['status'] for row in job['results']], ['completed', 'cancelled', 'cancelled'])
        self.assertEqual(self.analyzer.call_count, 2)
        self.assertEqual(self.client.get(job['results'][0]['result']['result_url']).status_code, 200)
        self.assertIsNone(aggregate_jobs.active())
        self.assertEqual(self.client.post(f'/api/aggregate/jobs/{job_id}/cancel').json()['status'], 'cancelled')
        aggregate_jobs.JOBS.clear()
        self.assertEqual(self.client.get(f'/api/aggregate/jobs/{job_id}').json()['status'], 'cancelled')

    def test_saved_history_export_and_delete_survive_worker_updates(self):
        entered, release = threading.Event(), threading.Event()
        normal = self.analyzer.side_effect

        def analyze(source, **options):
            if source.name == 'b.png':
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('测试等待超时')
            result = normal(source, **options)
            result['targets'] = [{'bbox': [0, 0, 10, 10], 'action': {'state': '低头', 'evidence': '<script>alert(1)</script>'}}]
            return result

        self.analyzer.side_effect = analyze
        response = self.client.post('/api/aggregate/batch', json={'filenames': ['a.png', 'b.png']})
        job_id = response.json()['job_id']
        try:
            self.assertTrue(entered.wait(5))
            rows = self.client.get('/api/aggregate/history').json()['records']
            self.assertEqual(len(rows), 1)
            record_id = rows[0]['id']
            opened = self.client.get(f'/api/aggregate/history/{record_id}').json()
            self.assertEqual(opened['source'], 'a.png')
            exported = self.client.post('/api/aggregate/history/export', json={'record_ids': [record_id]})
            self.assertEqual(exported.status_code, 200)
            with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
                payload = json.loads(archive.read('results.json'))
                self.assertTrue(payload['records'][0]['result']['result_url'].startswith('data:image/png;base64,'))
                self.assertEqual(payload['records'][0]['result']['targets'][0]['action']['evidence'], '<script>alert(1)</script>')
                viewer = archive.read('viewer.html').decode('utf-8')
                embedded = viewer.split('<script id="embedded-results" type="application/json">')[1].split('</script>')[0]
                self.assertEqual(json.loads(embedded), payload)
                self.assertNotIn('<script>alert(1)</script>', viewer)
                self.assertIn("fetch('./results.json'", viewer)
            deleted = self.client.post('/api/aggregate/history/delete', json={'record_ids': [record_id]})
            self.assertEqual(deleted.json()['deleted'], [record_id])
            self.assertEqual(self.client.get(f'/api/aggregate/history/{record_id}').status_code, 404)
        finally:
            release.set()
        self.wait(job_id)
        rows = self.client.get('/api/aggregate/history').json()['records']
        self.assertEqual([row['source'] for row in rows], ['b.png'])
        self.assertEqual(self.analyzer.call_count, 2)
        self.assertEqual(self.client.post('/api/aggregate/history/export', json={'record_ids': []}).status_code, 422)
        self.assertEqual(self.client.get('/api/aggregate/history/invalid-id').status_code, 404)
