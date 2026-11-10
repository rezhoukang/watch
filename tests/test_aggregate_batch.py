"""Batch queue, failure isolation and stable input/result snapshots."""
import tempfile
import io
import json
import zipfile
import base64
import random
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

import aggregate
import aggregate_batch
import aggregate_history
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

    def test_pending_image_folder_move_and_legacy_paths(self):
        metadata = self.root / 'image-folders.json'
        metadata.write_text(json.dumps({'a.png': '旧目录/原名.png'}), encoding='utf-8')
        self.assertIn('旧目录', self.client.get('/api/images').json()['folders'])
        response = self.client.post('/api/image-folders', json={'name': '新目录', 'parent': '旧目录'})
        self.assertEqual(response.status_code, 200)
        target = response.json()['path']
        self.assertEqual(self.client.post('/api/image-folders/move', json={
            'filenames': ['a.png', 'b.png'], 'target': target,
        }).status_code, 200)
        listing = self.client.get('/api/images').json()
        images = {image['filename']: image for image in listing['images']}
        self.assertEqual(images['a.png']['relative_path'], '旧目录/新目录/原名.png')
        self.assertEqual(images['b.png']['relative_path'], '旧目录/新目录/b.png')
        self.assertTrue((self.uploads / 'a.png').exists())
        before = metadata.read_text(encoding='utf-8')
        self.assertEqual(self.client.post('/api/image-folders/move', json={
            'filenames': ['a.png', 'missing.png'], 'target': '',
        }).status_code, 404)
        self.assertEqual(metadata.read_text(encoding='utf-8'), before)
        self.assertEqual(self.client.post('/api/image-folders/move', json={
            'filenames': ['a.png'], 'target': '',
        }).status_code, 200)
        listing = self.client.get('/api/images').json()
        self.assertIn(target, listing['folders'])
        moved = next(image for image in listing['images'] if image['filename'] == 'a.png')
        self.assertEqual(moved['display_name'], '原名.png')
        self.assertEqual(moved['relative_path'], '原名.png')
        self.assertEqual(self.client.post('/api/image-folders', json={'name': '../非法'}).status_code, 400)

    def test_history_rerun_uses_saved_sources_and_keeps_duplicate_names(self):
        first = self.wait(self.client.post('/api/aggregate/batch', json={'filenames': ['a.png']}).json()['job_id'])
        first_id = first['job_id'] + '-0'
        original = self.client.get('/api/aggregate/history/' + first_id).json()
        Image.new('RGB', (32, 24), 'green').save(self.uploads / 'a.png')
        second = self.wait(self.client.post('/api/aggregate/batch', json={'filenames': ['a.png']}).json()['job_id'])
        second_id = second['job_id'] + '-0'
        (self.uploads / 'a.png').unlink()
        folder = self.client.post('/api/aggregate/folders', json={'name': '重跑测试'}).json()['id']
        self.client.post('/api/aggregate/folders/move', json={'record_ids': [first_id, second_id], 'folder_id': folder})
        response = self.client.post('/api/aggregate/history/rerun', json={
            'record_ids': [first_id], 'folder_ids': [folder], 'multimodal': False,
        })
        self.assertEqual(response.status_code, 200)
        job = self.wait(response.json()['job_id'])
        self.assertEqual(job['completed_images'], 2)
        self.assertEqual([row['source'] for row in job['results']], ['a.png', 'a.png'])
        self.assertNotEqual(job['results'][0]['source_file'], job['results'][1]['source_file'])
        colors = []
        ids = []
        for index, row in enumerate(job['results']):
            path = aggregate_batch.directory(job['job_id']) / 'sources' / row['source_file']
            with Image.open(path) as image:
                colors.append(image.getpixel((0, 0)))
            ids.append(job['job_id'] + '-' + str(index))
        self.assertEqual(colors, [(255, 255, 255), (0, 128, 0)])
        self.assertEqual(self.client.get('/api/aggregate/history/' + first_id).json(), original)
        exported = self.client.post('/api/aggregate/history/export', json={'record_ids': ids})
        self.assertEqual(exported.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(exported.content)) as bundle:
            records = json.loads(bundle.read('results.json'))['records']
            self.assertNotEqual(records[0]['result']['source_url'], records[1]['result']['source_url'])
        self.assertEqual(self.client.post('/api/aggregate/history/rerun', json={}).status_code, 422)
        source = aggregate_batch.directory(first['job_id']) / 'sources' / 'a.png'
        source.unlink()
        self.assertEqual(self.client.post('/api/aggregate/history/rerun', json={'record_ids': [first_id]}).status_code, 404)

    def test_results_mirror_source_folders_and_rerun_keeps_snapshot_paths(self):
        metadata = self.root / 'image-folders.json'
        metadata.write_text(json.dumps({
            'a.png': 'monitor-pose/东教/101.png',
            'b.png': 'monitor-pose/西教/101.png',
        }), encoding='utf-8')
        job = self.wait(self.client.post('/api/aggregate/batch', json={
            'filenames': ['a.png', 'b.png', 'c.png'],
        }).json()['job_id'])
        self.assertEqual([row['source'] for row in job['results']], ['101.png', '101.png', 'c.png'])
        self.assertEqual(job['results'][0]['relative_path'], 'monitor-pose/东教/101.png')
        folders = self.client.get('/api/aggregate/folders').json()['folders']
        root = next(folder for folder in folders if folder['name'] == 'monitor-pose')
        east = next(folder for folder in folders if folder['name'] == '东教')
        west = next(folder for folder in folders if folder['name'] == '西教')
        self.assertEqual(east['parent_id'], root['id'])
        self.assertEqual(west['parent_id'], root['id'])
        self.assertEqual(east['record_ids'], [job['job_id'] + '-0'])
        self.assertEqual(west['record_ids'], [job['job_id'] + '-1'])
        metadata.write_text(json.dumps({'a.png': '后来移动/101.png'}), encoding='utf-8')
        (self.uploads / 'a.png').unlink()
        rerun = self.client.post('/api/aggregate/history/rerun', json={'folder_ids': [root['id']]})
        self.assertEqual(rerun.status_code, 200)
        rerun_job = self.wait(rerun.json()['job_id'])
        self.assertEqual({row['relative_path'] for row in rerun_job['results']},
                         {'monitor-pose/东教/101.png', 'monitor-pose/西教/101.png'})
        folders = self.client.get('/api/aggregate/folders').json()['folders']
        self.assertEqual(len(folders), 3)
        self.assertTrue(all(len(folder['record_ids']) == 2 for folder in folders if folder['parent_id']))
        exported = self.client.post('/api/aggregate/history/export', json={'folder_ids': [root['id']]})
        self.assertEqual(exported.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(exported.content)) as bundle:
            self.assertIn('monitor-pose.html', bundle.namelist())
            self.assertEqual(len(json.loads(bundle.read('monitor-pose.json'))['records']), 4)

    def test_legacy_results_are_classified_once_and_manual_moves_are_preserved(self):
        job = self.wait(self.client.post('/api/aggregate/batch', json={'filenames': ['a.png', 'b.png']}).json()['job_id'])
        manifest = aggregate_batch.directory(job['job_id']) / 'record.json'
        batch = json.loads(manifest.read_text(encoding='utf-8'))
        for row in batch['results']:
            row.pop('relative_path')
            row.pop('source_file')
        manifest.write_text(json.dumps(batch), encoding='utf-8')
        (self.root / 'image-folders.json').write_text(json.dumps({
            'a.png': '旧目录/子目录/a.png', 'b.png': '旧目录/子目录/b.png',
        }), encoding='utf-8')
        manual = self.client.post('/api/aggregate/folders', json={'name': '手动分类'}).json()['id']
        ids = [job['job_id'] + '-0', job['job_id'] + '-1']
        self.client.post('/api/aggregate/folders/move', json={'record_ids': [ids[1]], 'folder_id': manual})
        # Previously saved folder lists remain readable during migration.
        path = aggregate.RESULT_DIR / 'history-folders.json'
        state = json.loads(path.read_text(encoding='utf-8'))
        path.write_text(json.dumps(state['folders']), encoding='utf-8')
        folders = self.client.get('/api/aggregate/folders').json()['folders']
        self.assertEqual(next(folder for folder in folders if folder['id'] == manual)['record_ids'], [ids[1]])
        self.assertEqual(next(folder for folder in folders if folder['name'] == '子目录')['record_ids'], [ids[0]])
        self.client.post('/api/aggregate/folders/move', json={'record_ids': [ids[0]], 'folder_id': None})
        folders = self.client.get('/api/aggregate/folders').json()['folders']
        self.assertFalse(any(ids[0] in folder['record_ids'] for folder in folders))
        self.assertEqual(self.client.get('/api/aggregate/folders').json()['folders'], folders)

    def test_mirror_batch_flips_pixels_and_saves_two_independent_results(self):
        image = Image.new('RGB', (32, 24), 'blue')
        for x in range(16):
            for y in range(24):
                image.putpixel((x, y), (255, 0, 0))
        image.save(self.uploads / 'a.png')
        normal = self.analyzer.side_effect
        observed = []
        def analyze(source, **options):
            with Image.open(source) as current:
                observed.append((current.getpixel((0, 0)), current.getpixel((31, 0))))
            return normal(source, **options)
        self.analyzer.side_effect = analyze
        response = self.client.post('/api/mirror/batch', json={'filenames': ['a.png', 'a.png']})
        self.assertEqual(response.status_code, 200)
        job = self.wait(response.json()['job_id'])
        self.assertTrue(job['mirror'])
        self.assertEqual(job['completed_images'], 1)
        self.assertEqual(observed, [((255, 0, 0), (0, 0, 255)), ((0, 0, 255), (255, 0, 0))])
        self.assertTrue(all(not options['face'] and not options['pose'] and not options['multimodal'] and not options['mirror_selection'] for _, options in self.calls))
        row = job['results'][0]
        self.assertNotEqual(row['result']['result_url'], row['mirror_result']['result_url'])
        (self.uploads / 'a.png').unlink()
        for result in [row['result'], row['mirror_result']]:
            self.assertEqual(self.client.get(result['source_url']).status_code, 200)
            self.assertEqual(self.client.get(result['result_url']).status_code, 200)
        self.assertEqual(self.client.get('/api/aggregate/history').json()['records'], [])
        history = self.client.get('/api/mirror/history').json()['records']
        self.assertEqual(len(history), 1)
        self.assertEqual(self.client.get('/api/mirror/history/' + history[0]['id']).json()['mirror_result'], row['mirror_result'])
        aggregate_jobs.JOBS.clear()
        self.assertEqual(self.client.get('/api/mirror/history/' + history[0]['id']).status_code, 200)
        self.assertEqual(self.client.get('/api/mirror/history/not-a-record').status_code, 404)
        self.assertEqual(self.client.post('/api/mirror/batch', json={'filenames':['b.png'], 'multimodal':True}).status_code, 422)

    def test_mirror_failure_isolated_and_stop_prevents_second_inference(self):
        normal = self.analyzer.side_effect
        def analyze(source, **options):
            if source.name == 'mirror_0000.png':
                raise ValueError('模拟镜像检测失败')
            return normal(source, **options)
        self.analyzer.side_effect = analyze
        response = self.client.post('/api/mirror/batch', json={'filenames':['a.png','b.png']})
        job = self.wait(response.json()['job_id'])
        self.assertEqual((job['completed_images'], job['failed_images']), (1, 1))
        self.assertEqual(self.client.get('/api/mirror/history/' + job['job_id'] + '-0').status_code, 404)
        entered, release = threading.Event(), threading.Event()
        def blocking(source, **options):
            entered.set()
            release.wait(5)
            return normal(source, **options)
        self.analyzer.side_effect = blocking
        self.analyzer.reset_mock()
        response = self.client.post('/api/mirror/batch', json={'filenames':['c.png']})
        self.assertTrue(entered.wait(3))
        active = self.client.get('/api/mirror/active').json()['job']
        self.assertEqual(active['job_id'], response.json()['job_id'])
        self.client.post('/api/aggregate/jobs/' + active['job_id'] + '/cancel')
        release.set()
        stopped = self.wait(active['job_id'])
        self.assertEqual(stopped['status'], 'cancelled')
        self.assertEqual(stopped['completed_images'], 0)
        self.assertEqual(self.analyzer.call_count, 1)

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

    def test_folder_assignment_persists_and_names_export_files(self):
        response = self.client.post('/api/aggregate/batch', json={'filenames': ['a.png', 'b.png']})
        job_id = response.json()['job_id']
        self.wait(job_id)
        ids = [f'{job_id}-0', f'{job_id}-1']
        name = "课堂 A's"
        folder = self.client.post('/api/aggregate/folders', json={'name': name}).json()
        self.assertEqual(self.client.post('/api/aggregate/folders', json={'name': name}).status_code, 400)
        for invalid in ['../bad', 'CON', 'name.', ' ']:
            self.assertEqual(self.client.post('/api/aggregate/folders', json={'name': invalid}).status_code, 400)
        moved = self.client.post('/api/aggregate/folders/move', json={'record_ids': ids, 'folder_id': folder['id']})
        self.assertEqual(moved.status_code, 200)
        self.assertEqual(self.client.get('/api/aggregate/folders').json()['folders'][0]['record_ids'], ids)
        bad_move = self.client.post('/api/aggregate/folders/move', json={'record_ids': [ids[0], 'invalid'], 'folder_id': None})
        self.assertEqual(bad_move.status_code, 404)
        self.assertEqual(self.client.get('/api/aggregate/folders').json()['folders'][0]['record_ids'], ids)
        exported = self.client.post('/api/aggregate/history/export', json={'folder_ids': [folder['id']]})
        self.assertEqual(exported.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            self.assertEqual(set(archive.namelist()), {name+'.json', name+'.html'})
            payload = json.loads(archive.read(name+'.json'))
            self.assertEqual(payload['title'], name)
            self.assertEqual(len(payload['records']), 2)
            viewer = archive.read(name+'.html').decode('utf-8')
            self.assertIn('./%E8%AF%BE%E5%A0%82%20A%27s.json', viewer)
        self.client.post('/api/aggregate/folders/move', json={'record_ids': [ids[0]], 'folder_id': None})
        self.assertEqual(self.client.get('/api/aggregate/folders').json()['folders'][0]['record_ids'], ids[1:])
        self.client.post('/api/aggregate/history/delete', json={'record_ids': ids[1:]})
        self.assertEqual(self.client.post('/api/aggregate/history/export', json={'folder_ids': [folder['id']]}).status_code, 400)

    def test_nested_folders_reject_cycles_and_export_descendant_results(self):
        job_id = self.client.post('/api/aggregate/batch', json={'filenames': ['a.png']}).json()['job_id']
        self.wait(job_id)
        parent = self.client.post('/api/aggregate/folders', json={'name': 'Parent'}).json()['id']
        child = self.client.post('/api/aggregate/folders', json={'name': 'Child', 'parent_id': parent}).json()['id']
        self.client.post('/api/aggregate/folders/move', json={'record_ids': [f'{job_id}-0'], 'folder_id': child})
        for target in (parent, child):
            response = self.client.post('/api/aggregate/folders/move', json={'folder_ids': [parent], 'folder_id': target})
            self.assertEqual(response.status_code, 400)
        exported = self.client.post('/api/aggregate/history/export', json={'folder_ids': [parent]})
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            self.assertEqual(len(json.loads(archive.read('Parent.json'))['records']), 1)
        moved = self.client.post('/api/aggregate/folders/move', json={'folder_ids': [child], 'folder_id': None})
        self.assertEqual(moved.status_code, 200)
        restored = self.client.get('/api/aggregate/folders').json()['folders']
        self.assertIsNone(next(row for row in restored if row['id'] == child)['parent_id'])

    def test_delete_mixed_records_and_nested_folders_is_retryable(self):
        job_id = self.client.post('/api/aggregate/batch', json={'filenames': ['a.png', 'b.png', 'c.png']}).json()['job_id']
        self.wait(job_id)
        parent = self.client.post('/api/aggregate/folders', json={'name': 'Parent'}).json()['id']
        child = self.client.post('/api/aggregate/folders', json={'name': 'Child', 'parent_id': parent}).json()['id']
        empty = self.client.post('/api/aggregate/folders', json={'name': 'Empty', 'parent_id': child}).json()['id']
        keep = self.client.post('/api/aggregate/folders', json={'name': 'Keep'}).json()['id']
        self.client.post('/api/aggregate/folders/move', json={'record_ids': [f'{job_id}-0'], 'folder_id': child})
        payload = {'record_ids': [f'{job_id}-1'], 'folder_ids': [parent, child]}
        # A failed folder-metadata write must not resurrect records at the root.
        with patch.object(aggregate_history, '_save_folders', side_effect=OSError('模拟写入失败')):
            with self.assertRaises(OSError):
                aggregate_history.delete(payload['record_ids'], payload['folder_ids'])
        self.assertEqual([row['source'] for row in self.client.get('/api/aggregate/history').json()['records']], ['c.png'])
        response = self.client.post('/api/aggregate/history/delete', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()['deleted_folders']), {parent, child, empty})
        self.assertEqual([row['id'] for row in self.client.get('/api/aggregate/folders').json()['folders']], [keep])
        self.assertEqual(self.client.get(f'/api/aggregate/history/{job_id}-0').status_code, 404)
        repeated = self.client.post('/api/aggregate/history/delete', json=payload).json()
        self.assertEqual(set(repeated['missing_folders']), {parent, child})
        self.assertEqual(repeated['deleted_folders'], [])
        self.assertEqual(self.client.post('/api/aggregate/history/delete', json={}).status_code, 422)

    def test_folder_upload_retains_relative_paths_without_filename_collisions(self):
        data = (self.uploads / 'a.png').read_bytes()
        uploaded = []
        for path in ('Class/Row1/same.png', 'Class/Row2/same.png'):
            response = self.client.post('/api/upload-folder-image', data={'relative_path': path}, files={'file': ('same.png', data, 'image/png')})
            self.assertEqual(response.status_code, 200)
            uploaded.append(response.json()['filename'])
        self.assertNotEqual(*uploaded)
        listed = self.client.get('/api/images').json()['images']
        paths = {row['relative_path'] for row in listed if row['filename'] in uploaded}
        self.assertEqual(paths, {'Class/Row1/same.png', 'Class/Row2/same.png'})
        for path in ('../x.png', '/x.png', 'Class/../x.png', 'Class/a.txt', 'C:/x.png'):
            self.assertEqual(self.client.post('/api/upload-folder-image', data={'relative_path': path}, files={'file': ('x.png', data)}).status_code, 400)
        self.assertEqual(self.client.post('/api/upload-folder-image', data={'relative_path': 'Class/bad.png'}, files={'file': ('bad.png', b'bad')}).status_code, 400)

    def test_compressed_export_resizes_images_and_preserves_target_data(self):
        image = Image.frombytes('RGB', (2200, 1200), random.Random(0).randbytes(2200 * 1200 * 3))
        image.save(self.uploads / 'a.png')
        original_bytes = (self.uploads / 'a.png').read_bytes()
        normal = self.analyzer.side_effect

        def analyze(source, **options):
            result = normal(source, **options)
            result.update(image_width=2200, image_height=1200,
                          targets=[{'bbox': [100, 200, 300, 400], 'action': {'state': '低头'}}])
            return result

        self.analyzer.side_effect = analyze
        job_id = self.client.post('/api/aggregate/batch', json={'filenames': ['a.png']}).json()['job_id']
        self.wait(job_id)
        ids = [f'{job_id}-0']
        original = self.client.post('/api/aggregate/history/export', json={'record_ids': ids})
        for mode, edge in [('balanced', 1600), ('compact', 1280)]:
            compressed = self.client.post('/api/aggregate/history/export', json={'record_ids': ids, 'compression': mode})
            self.assertEqual(compressed.status_code, 200)
            self.assertLess(len(compressed.content), len(original.content) / 2)
            with zipfile.ZipFile(io.BytesIO(compressed.content)) as archive:
                result = json.loads(archive.read('results.json'))['records'][0]['result']
                self.assertTrue(result['source_url'].startswith('data:image/jpeg;base64,'))
                with Image.open(io.BytesIO(base64.b64decode(result['source_url'].split(',')[1]))) as exported_image:
                    self.assertEqual(exported_image.width, edge)
                self.assertEqual(result['image_width'], 2200)
                self.assertEqual(result['targets'][0]['bbox'], [100, 200, 300, 400])
                self.assertEqual(result['targets'][0]['action']['state'], '低头')
        self.assertEqual((self.uploads / 'a.png').read_bytes(), original_bytes)
        self.assertEqual(self.client.post('/api/aggregate/history/export', json={'record_ids': ids, 'compression': 'bad'}).status_code, 422)

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
