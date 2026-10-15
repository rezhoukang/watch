"""Face library management and 1:N recognition tests (mocked model)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

import app
import face_recognition
import head_cut_faces
import head_cuts
import vision


class FaceRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # 把库目录指到临时目录，避免污染真实输出
        for module, field, value in (
            (face_recognition, 'LIBRARY_DIR', self.root / 'face_library'),
            (face_recognition, 'LIBRARY_JSON', self.root / 'face_library' / 'library.json'),
            (face_recognition, 'LIBRARY_NPZ', self.root / 'face_library' / 'embeddings.npz'),
        ):
            mocked = patch.object(module, field, value); mocked.start(); self.addCleanup(mocked.stop)
        self.client = TestClient(app.app)

    def test_empty_library_lists_zero(self):
        response = self.client.get('/api/face-library')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['entries'], [])

    def test_match_threshold_decision(self):
        """在未建库时 match 应返回 not matched。"""
        import numpy as np
        result = face_recognition.match(np.ones(512, dtype=np.float32), threshold=0.4)
        self.assertFalse(result['matched'])
        self.assertIsNone(result['name'])

    def test_delete_missing_entry(self):
        self.assertFalse(face_recognition.delete_entry('9999'))

    def test_add_entry_requires_face(self):
        """无人脸照片应被拒绝。"""
        import numpy as np
        blank = np.full((64, 64, 3), 128, dtype=np.uint8)
        with patch.object(face_recognition, '_get_detector') as detector_get:
            detector = Mock()
            detector.detect.return_value = []
            detector_get.return_value = detector
            with self.assertRaises(ValueError):
                face_recognition.add_entry('测试', blank)


class RecognitionTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = self.root / 'scrfd.onnx'; model.touch()
        for module, field, value in (
            (head_cuts, 'HEAD_CUT_DIR', self.root / 'output'),
            (head_cut_faces, 'MODEL_PATH', model),
            (face_recognition, 'LIBRARY_DIR', self.root / 'face_library'),
            (face_recognition, 'LIBRARY_JSON', self.root / 'face_library' / 'library.json'),
            (face_recognition, 'LIBRARY_NPZ', self.root / 'face_library' / 'embeddings.npz'),
        ):
            mocked = patch.object(module, field, value); mocked.start(); self.addCleanup(mocked.stop)
        source = self.root / 'image.png'
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

    def _seed_faces(self):
        """为批次注入一条已检出的 faces 结果（含 landmarks），并 mock 识别模型。"""
        folder = head_cuts.run_dir(self.run_id)
        face = {
            'face_id': 'HC001-001_F01', 'parent_crop_id': 'HC001-001',
            'confidence': .9, 'bbox': [2, 3, 12, 16],
            'landmarks': [[3, 5], [8, 5], [5, 9], [4, 11], [7, 11]],
            'file': 'faces/job/face.png', 'width': 10, 'height': 13,
        }
        # 保存真实裁剪图
        crop_img = Image.new('RGB', (14, 17), '#123456')
        target = folder / 'faces' / 'job' / 'face.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        crop_img.save(target)
        record = {
            'job_id': 'job', 'run_id': self.run_id, 'status': 'done',
            'total': 1, 'progress': 1, 'heads_with_faces': 1, 'face_count': 1,
            'results': {'HC001-001': {'crop_id': 'HC001-001', 'status': 'detected',
                                      'face_count': 1, 'faces': [face]}},
            'errors': [],
        }
        head_cut_faces._save(self.run_id, record)

    def test_recognize_empty_library_rejected(self):
        self._seed_faces()
        response = self.client.post(f'/api/head-cuts/{self.run_id}/recognize', json={'threshold': .4})
        self.assertEqual(response.status_code, 400)
        self.assertIn('人脸库为空', response.json()['detail'])

    def test_recognize_writes_results(self):
        self._seed_faces()
        # 注入一个模拟的库：直接写一个 entry 和 embedding
        import numpy as np
        face_recognition.add_entry = Mock(side_effect=lambda name, img, source='upload': {'id': '0001', 'name': name})
        face_recognition.list_entries = Mock(return_value=[{'id': '0001', 'name': '张三'}])
        # mock 特征提取 + 匹配 + 质量筛选通过
        face_recognition.extract_embedding = Mock(return_value=np.ones(512, dtype=np.float32) / np.sqrt(512))
        face_recognition.assess_quality = Mock(return_value=([], {'sharpness': 100}))
        face_recognition.match = Mock(return_value={
            'matched': True, 'entry_id': '0001', 'name': '张三', 'similarity': 0.95})
        face_recognition.match_top_k = Mock(return_value=[
            {'entry_id': '0001', 'name': '张三', 'similarity': 0.95}])
        with patch.object(head_cut_faces.threading, 'Thread'):
            response = self.client.post(f'/api/head-cuts/{self.run_id}/recognize', json={'threshold': .4})
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()['job_id']
        self.addCleanup(vision.JOBS.pop, job_id, None)
        # 直接跑后台任务
        batch = head_cuts.read_run(self.run_id)
        crops = [{'crop_id': 'HC001-001', 'faces': batch['crops'][0]['face_detection']['faces']}]
        record = json.loads((head_cuts.run_dir(self.run_id) / 'recognition.json').read_text(encoding='utf-8'))
        head_cut_faces._run_recognition(self.run_id, job_id, crops, record, 0.4)
        # 验证结果写入
        rec = json.loads((head_cuts.run_dir(self.run_id) / 'recognition.json').read_text(encoding='utf-8'))
        self.assertEqual(rec['status'], 'done')
        self.assertEqual(rec['matched'], 1)
        self.assertEqual(rec['results']['HC001-001']['faces'][0]['match']['name'], '张三')


if __name__ == '__main__':
    unittest.main()
