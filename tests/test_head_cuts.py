import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import app
import head_cuts
import vision
from web import common


class HeadCutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for module, field, value in ((head_cuts, 'HEAD_CUT_DIR', self.root/'output'), (common, 'UPLOAD_DIR', self.root)):
            mocked = patch.object(module, field, value)
            mocked.start(); self.addCleanup(mocked.stop)
        for name in ('a教室.png', 'b教室.png'):
            Image.new('RGB', (100, 80), '#abcdef').save(self.root/name)
        self.client = TestClient(app.app)
        self.prediction = {'detections': [
            {'bbox': [-1.5, 2.1, 20.2, 30.8], 'class_name': 'head', 'confidence': .8},
            {'bbox': [50, 40, 80, 70], 'class_name': 'phone', 'confidence': .7}]}

    def start(self):
        with patch.object(common, '_require_head_detection'), patch.object(head_cuts.threading, 'Thread'):
            response = self.client.post('/api/head-cuts', json={'filenames': ['a教室.png', 'b教室.png']})
        self.assertEqual(response.status_code, 200, response.text)
        run_id = response.json()['run_id']
        self.addCleanup(vision.JOBS.pop, run_id, None)
        return run_id

    def finish(self, run_id):
        with patch.object(head_cuts.head_fusion, 'detect_image', return_value=self.prediction) as detect:
            head_cuts._run_batch(run_id)
        self.assertEqual(detect.call_args.kwargs, {'rtdetr_base': False, 'merge_iou': .1, 'merge_confidence': .55})

    def test_batch_crops_every_class_uses_snapshot_and_preserves_marks(self):
        run_id = self.start()
        # Replacing a pending source must not change this batch's frozen image.
        Image.new('RGB', (100, 80), 'red').save(self.root/'a教室.png')
        self.finish(run_id)
        response = self.client.get(f'/api/head-cuts/{run_id}')
        record = response.json()
        self.assertEqual(record['status'], 'done')
        self.assertEqual(record['crop_count'], 4)
        self.assertEqual([v['class_name'] for v in record['crops']], ['head', 'phone', 'head', 'phone'])
        crop = record['crops'][0]
        self.assertEqual(crop['crop_bbox'], [0, 2, 21, 31])
        with Image.open(head_cuts.run_dir(run_id)/crop['raw_file']) as image:
            self.assertEqual(image.size, (21, 29))
            self.assertEqual(image.getpixel((2, 2)), (171, 205, 239))
        with Image.open(head_cuts.run_dir(run_id)/crop['marked_file']) as image:
            self.assertGreater(image.height, 29)
        mark = self.client.post(f'/api/head-cuts/{run_id}/marks', json={'marks': {crop['crop_id']: '已核对'}})
        self.assertEqual(mark.status_code, 200)
        self.assertEqual(head_cuts.read_run(run_id)['crops'][0]['mark'], '已核对')
        self.assertEqual(self.client.get(f'/api/head-cuts/{run_id}/files/'+crop['raw_file']).status_code, 200)
        self.assertEqual(self.client.get(f'/api/head-cuts/{run_id}/download').status_code, 200)
        with zipfile.ZipFile(head_cuts.HEAD_CUT_DIR/f'{run_id}.zip') as archive:
            self.assertEqual(archive.testzip(), None)
            self.assertIn('marks.json', archive.namelist())
            self.assertEqual(json.loads(archive.read('marks.json'))[crop['crop_id']], '已核对')

    def test_invalid_batch_does_not_create_jobs_or_files(self):
        for names in ([], ['../a教室.png'], ['missing.png']):
            response = self.client.post('/api/head-cuts', json={'filenames': names})
            self.assertIn(response.status_code, (400, 404, 422))
        self.assertFalse(head_cuts.HEAD_CUT_DIR.exists())
        response = self.client.post('/api/head-cuts', json={'filenames': ['a教室.png'], 'merge_iou': 1.1})
        self.assertEqual(response.status_code, 422)

    def test_cancel_interruption_and_file_path_checks(self):
        run_id = self.start()
        vision.JOBS[run_id]['cancel'] = True
        head_cuts._run_batch(run_id)
        self.assertEqual(head_cuts.read_run(run_id)['status'], 'cancelled')
        self.assertEqual(head_cuts.read_run(run_id)['crop_count'], 0)
        self.assertEqual(self.client.get(f'/api/head-cuts/{run_id}/files/%2e%2e%2foutside.png').status_code, 404)
        self.assertEqual(self.client.get('/api/head-cuts/not-a-run').status_code, 400)
        self.assertEqual(self.client.post(f'/api/head-cuts/{run_id}/marks', json={'marks': {'unknown': 'x'}}).status_code, 400)
        other = self.start()
        vision.JOBS.pop(other)
        self.assertEqual(head_cuts.read_run(other)['status'], 'interrupted')

    def test_per_image_failure_keeps_other_successful_crops(self):
        run_id = self.start()
        with patch.object(head_cuts.head_fusion, 'detect_image', side_effect=[RuntimeError('test failure'), self.prediction]):
            head_cuts._run_batch(run_id)
        record = head_cuts.read_run(run_id)
        self.assertEqual(record['status'], 'done_with_errors')
        self.assertEqual(record['crop_count'], 2)
        self.assertEqual(record['errors'][0]['source'], 'a教室.png')


if __name__ == '__main__':
    unittest.main()
