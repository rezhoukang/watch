import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

import app
import head_cuts
import head_cut_faces
import vision


class HeadCutFaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = self.root/'scrfd.onnx'; model.touch()
        for module, field, value in ((head_cuts, 'HEAD_CUT_DIR', self.root/'output'), (head_cut_faces, 'MODEL_PATH', model)):
            mocked = patch.object(module, field, value); mocked.start(); self.addCleanup(mocked.stop)
        source = self.root/'image.png'
        Image.new('RGB', (100, 80), '#abcdef').save(source)
        with patch.object(head_cuts.threading, 'Thread'):
            self.run_id = head_cuts.start_batch([source])
        self.addCleanup(vision.JOBS.pop, self.run_id, None)
        prediction = {'detections': [
            {'bbox': [0, 2, 21, 31], 'class_name': 'head', 'confidence': .9},
            {'bbox': [30, 20, 60, 50], 'class_name': 'head', 'confidence': .8}]}
        with patch.object(head_cuts.head_fusion, 'detect_image', return_value=prediction):
            head_cuts._run_batch(self.run_id)
        self.client = TestClient(app.app)

    def start_faces(self):
        with patch.object(head_cut_faces.threading, 'Thread'):
            response = self.client.post(f'/api/head-cuts/{self.run_id}/faces', json={'confidence': .5})
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()['job_id']
        self.addCleanup(vision.JOBS.pop, job_id, None)
        return job_id

    def finish_faces(self, job_id, detector):
        folder = head_cuts.run_dir(self.run_id)
        record = json.loads((folder/'face_detection.json').read_text(encoding='utf-8'))
        with patch.object(head_cut_faces, '_get_detector', return_value=detector):
            head_cut_faces._run_detection(self.run_id, job_id, head_cuts.read_run(self.run_id)['crops'], record)

    def test_faces_keep_parent_ids_original_pixels_coordinates_marks_and_zip(self):
        before = head_cuts.read_run(self.run_id)
        parent_id = before['crops'][0]['crop_id']
        head_cuts.save_marks(self.run_id, {parent_id: '人工备注'})
        head_cuts.build_archive(self.run_id)
        job_id = self.start_faces()
        detector = Mock()
        face = {'bbox': [1, 2, 11, 14], 'confidence': .9,
                'landmarks': [[3, 4], [8, 4], [5, 8], [4, 10], [7, 10]]}
        detector.detect.side_effect = [[face, {**face, 'bbox': [12, 2, 20, 14], 'confidence': .8}], []]
        self.finish_faces(job_id, detector)
        record = self.client.get(f'/api/head-cuts/{self.run_id}').json()
        self.assertEqual(record['crop_count'], 2)
        self.assertEqual(record['face_detection']['status'], 'done')
        self.assertEqual(record['face_detection']['heads_with_faces'], 1)
        self.assertEqual(record['face_detection']['face_count'], 2)
        parent, empty = record['crops']
        self.assertEqual(parent['crop_id'], parent_id)
        self.assertEqual(parent['mark'], '人工备注')
        self.assertEqual(empty['face_detection']['status'], 'no_face')
        self.assertEqual(detector.detect.call_args_list[0].args[0].shape, (29, 21, 3))
        first = parent['face_detection']['faces'][0]
        self.assertEqual(first['parent_crop_id'], parent_id)
        self.assertEqual(first['source_bbox'], [1, 4, 11, 16])
        self.assertTrue(first['face_id'].startswith(parent_id))
        self.assertEqual(self.client.get(f'/api/head-cuts/{self.run_id}/files/'+first['file']).status_code, 200)
        with Image.open(head_cuts.run_dir(self.run_id)/first['file']) as image:
            self.assertEqual(image.size, (10, 12))
            self.assertEqual(image.getpixel((1, 1)), (171, 205, 239))
        with zipfile.ZipFile(head_cuts.build_archive(self.run_id)) as archive:
            self.assertIn('face_detection.json', archive.namelist())
            self.assertIn(first['file'], archive.namelist())

    def test_rejects_parallel_tasks_and_invalid_confidence(self):
        self.start_faces()
        self.assertEqual(self.client.post(f'/api/head-cuts/{self.run_id}/faces', json={}).status_code, 400)
        with self.assertRaises(ValueError):
            head_cuts.build_archive(self.run_id)
        for confidence in (0, 1.1, -.1):
            self.assertEqual(self.client.post(f'/api/head-cuts/{self.run_id}/faces', json={'confidence': confidence}).status_code, 422)

    def test_cancel_interruption_and_individual_errors_preserve_heads(self):
        job_id = self.start_faces()
        vision.JOBS[job_id]['cancel'] = True
        detector = Mock()
        self.finish_faces(job_id, detector)
        detector.detect.assert_not_called()
        self.assertEqual(head_cuts.read_run(self.run_id)['face_detection']['status'], 'cancelled')
        job_id = self.start_faces()
        detector.detect.side_effect = [RuntimeError('broken crop'), []]
        self.finish_faces(job_id, detector)
        record = head_cuts.read_run(self.run_id)
        self.assertEqual(record['face_detection']['status'], 'done_with_errors')
        self.assertEqual(record['crops'][0]['face_detection']['status'], 'error')
        self.assertEqual(record['crops'][1]['face_detection']['status'], 'no_face')
        self.assertEqual(record['crop_count'], 2)
        job_id = self.start_faces()
        vision.JOBS.pop(job_id)
        self.assertEqual(head_cuts.read_run(self.run_id)['face_detection']['status'], 'interrupted')


if __name__ == '__main__':
    unittest.main()
