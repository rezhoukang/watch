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
            (face_recognition, 'PHOTO_DIR', self.root / 'face_library' / 'photos'),
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

    def test_add_entry_stores_photo_and_serves_it(self):
        """入库时保存对齐后的登记照，可通过接口取回；删除时一并清理。"""
        import numpy as np
        image = np.random.default_rng(3).integers(0, 255, (120, 120, 3), dtype=np.uint8)
        face = {'bbox': [20, 20, 100, 110], 'confidence': .95,
                'landmarks': [[40, 50], [80, 50], [60, 62], [45, 85], [75, 85]]}
        detector = Mock(); detector.detect.return_value = [face]
        embedding = np.ones(512, dtype=np.float32) / np.sqrt(512)
        with patch.object(face_recognition, '_get_detector', return_value=detector), \
                patch.object(face_recognition, 'extract_embedding', return_value=embedding):
            created = face_recognition.add_entry('测试同学', image)
        entry_id = created['id']
        self.assertIsNotNone(face_recognition.photo_file(entry_id))
        response = self.client.get(f'/api/face-library/{entry_id}/photo')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers['content-type'], 'image/jpeg')
        # 列表里也带上可用的照片地址
        entries = self.client.get('/api/face-library').json()['entries']
        self.assertEqual(entries[0]['id'], entry_id)
        self.assertTrue(face_recognition.delete_entry(entry_id))
        self.assertIsNone(face_recognition.photo_file(entry_id))
        self.assertEqual(self.client.get(f'/api/face-library/{entry_id}/photo').status_code, 404)

    def test_photo_endpoint_rejects_unknown_or_illegal_id(self):
        self.assertEqual(self.client.get('/api/face-library/9999/photo').status_code, 404)
        self.assertEqual(self.client.get('/api/face-library/abc/photo').status_code, 404)
        self.assertIsNone(face_recognition.photo_path('../../etc/passwd'))


class RecognitionTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = self.root / 'scrfd.onnx'; model.touch()
        for module, field, value in (
            (head_cuts, 'HEAD_CUT_DIR', self.root / 'output'),
            (head_cut_faces, 'MODEL_PATH', model),
            (face_recognition, 'MODEL_PATH_RECOGNITION', model),
            (face_recognition, 'LIBRARY_DIR', self.root / 'face_library'),
            (face_recognition, 'LIBRARY_JSON', self.root / 'face_library' / 'library.json'),
            (face_recognition, 'LIBRARY_NPZ', self.root / 'face_library' / 'embeddings.npz'),
            (face_recognition, 'PHOTO_DIR', self.root / 'face_library' / 'photos'),
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
        # TestClient 的 blocking portal 在首次请求时才创建线程；识别接口的首次请求发生在
        # threading.Thread 被 mock 的窗口内，不预热会永久阻塞。
        self.client.get('/api/__portal_warmup__')

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
        crops = [{'crop_id': 'HC001-001', 'raw_file': batch['crops'][0]['raw_file'],
                  'faces': batch['crops'][0]['face_detection']['faces']}]
        record = json.loads((head_cuts.run_dir(self.run_id) / 'recognition.json').read_text(encoding='utf-8'))
        head_cut_faces._run_recognition(self.run_id, job_id, crops, record, 0.4)
        # 验证结果写入
        rec = json.loads((head_cuts.run_dir(self.run_id) / 'recognition.json').read_text(encoding='utf-8'))
        self.assertEqual(rec['status'], 'done')
        self.assertEqual(rec['matched'], 1)
        self.assertEqual(rec['results']['HC001-001']['faces'][0]['match']['name'], '张三')

    def test_recognize_reads_hc_crop_not_face_crop(self):
        """识别的 bbox/landmarks 相对 HC 裁剪图，必须用 HC 裁剪图做质量筛选与对齐。

        若错用人脸小图（faces/…png），坐标系不一致会导致取样区域和对齐均错位。
        """
        self._seed_faces()
        import numpy as np
        face_recognition.list_entries = Mock(return_value=[{'id': '0001', 'name': '张三'}])
        seen = {}
        face_recognition.assess_quality = Mock(
            side_effect=lambda image, face, **kwargs: (seen.setdefault('quality', image.shape), ([], {'sharpness': 100}))[1])
        face_recognition.extract_embedding = Mock(
            side_effect=lambda image, landmarks: (seen.setdefault('embed', (image.shape, landmarks)),
                                                  np.ones(512, dtype=np.float32) / np.sqrt(512))[1])
        face_recognition.match = Mock(return_value={
            'matched': True, 'entry_id': '0001', 'name': '张三', 'similarity': 0.95})
        face_recognition.match_top_k = Mock(return_value=[])
        with patch.object(head_cut_faces.threading, 'Thread'):
            response = self.client.post(f'/api/head-cuts/{self.run_id}/recognize', json={'threshold': .4})
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()['job_id']
        self.addCleanup(vision.JOBS.pop, job_id, None)
        batch = head_cuts.read_run(self.run_id)
        crops = [{'crop_id': 'HC001-001', 'raw_file': batch['crops'][0]['raw_file'],
                  'faces': batch['crops'][0]['face_detection']['faces']}]
        record = json.loads((head_cuts.run_dir(self.run_id) / 'recognition.json').read_text(encoding='utf-8'))
        head_cut_faces._run_recognition(self.run_id, job_id, crops, record, 0.4)
        # HC 裁剪图为 21x29（prediction bbox [0,2,21,31]）；人脸小图为 10x13
        self.assertEqual(seen['quality'], (29, 21, 3))
        self.assertEqual(seen['embed'][0], (29, 21, 3))
        # landmarks 应保持 HC 坐标系原值，未被平移
        self.assertEqual(seen['embed'][1], [[3, 5], [8, 5], [5, 9], [4, 11], [7, 11]])
        rec = json.loads((head_cuts.run_dir(self.run_id) / 'recognition.json').read_text(encoding='utf-8'))
        self.assertEqual(rec['results']['HC001-001']['faces'][0]['face_pixels'], 10)

    def test_recognize_records_configurable_quality_gates(self):
        """质量门槛可调：不再硬编码 0.8，且写入记录 config 以便追溯。"""
        self._seed_faces()
        face_recognition.list_entries = Mock(return_value=[{'id': '0001', 'name': '张三'}])
        payload = {'threshold': .35, 'min_confidence': .55, 'min_size': 12, 'min_sharpness': 0}
        with patch.object(head_cut_faces.threading, 'Thread'):
            response = self.client.post(f'/api/head-cuts/{self.run_id}/recognize', json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.addCleanup(vision.JOBS.pop, response.json()['job_id'], None)
        config = json.loads((head_cuts.run_dir(self.run_id) / 'recognition.json').read_text(encoding='utf-8'))['config']
        for key, value in payload.items():
            self.assertEqual(config[key], value, f'{key} 应原样写入 config')

    def test_recognize_rejects_out_of_range_gates(self):
        self._seed_faces()
        face_recognition.list_entries = Mock(return_value=[{'id': '0001', 'name': '张三'}])
        for bad in ({'min_confidence': 1.5}, {'min_confidence': -0.1}, {'min_size': 0}, {'min_sharpness': -1}):
            response = self.client.post(f'/api/head-cuts/{self.run_id}/recognize', json={'threshold': .4, **bad})
            self.assertEqual(response.status_code, 422, f'{bad} 应被拒绝：{response.text}')

    def test_recognize_resolves_duplicate_identity_one_to_one(self):
        """两张脸都指向同一个人时高分优先，另一张顺延到它的下一个候选。"""
        import numpy as np
        self._seed_faces()
        face_recognition.list_entries = Mock(return_value=[{'id': '0001', 'name': '张三'}])
        face_recognition.extract_embedding = Mock(return_value=np.ones(512, dtype=np.float32) / np.sqrt(512))
        face_recognition.assess_quality = Mock(return_value=([], {'sharpness': 100}))
        # 第 1 张脸最高分 .55；第 2 张脸的候选分更高（.62）→ 它应该抢走 0001
        face_recognition.match_top_k = Mock(side_effect=[
            [self._candidate('0001', '张三', .55), self._candidate('0002', '李四', .45)],
            [self._candidate('0001', '张三', .62)],
        ])
        batch = head_cuts.read_run(self.run_id)
        raw = batch['crops'][0]['raw_file']

        def face(face_id):
            return {'face_id': face_id, 'confidence': .9, 'bbox': [2, 3, 12, 16],
                    'landmarks': [[3, 5], [8, 5], [5, 9], [4, 11], [7, 11]]}

        crops = [{'crop_id': 'HC001-001', 'raw_file': raw, 'faces': [face('HC001-001_F01')]},
                 {'crop_id': 'HC001-002', 'raw_file': raw, 'faces': [face('HC001-002_F01')]}]
        with patch.object(head_cut_faces.threading, 'Thread'):
            response = self.client.post(f'/api/head-cuts/{self.run_id}/recognize', json={'threshold': .4})
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()['job_id']
        self.addCleanup(vision.JOBS.pop, job_id, None)
        record = json.loads((head_cuts.run_dir(self.run_id) / 'recognition.json').read_text(encoding='utf-8'))
        head_cut_faces._run_recognition(self.run_id, job_id, crops, record, 0.4)
        result = json.loads((head_cuts.run_dir(self.run_id) / 'recognition.json').read_text(encoding='utf-8'))
        loser = result['results']['HC001-001']['faces'][0]
        winner = result['results']['HC001-002']['faces'][0]
        self.assertEqual(winner['match']['entry_id'], '0001')
        self.assertEqual(winner['match']['rank'], 1)
        # 被抢的那张顺延到第 2 候选，并记录是谁抢走的
        self.assertEqual(loser['status'], 'matched')
        self.assertEqual(loser['match']['entry_id'], '0002')
        self.assertEqual(loser['match']['rank'], 2)
        self.assertEqual(loser['conflict']['skipped'][0]['taken_by'], 'HC001-002_F01')
        self.assertEqual(result['matched'], 2)
        self.assertEqual(result['reassigned'], 1)
        self.assertEqual(result['conflicts'], 1)

    @staticmethod
    def _candidate(entry_id, name, similarity):
        return {'entry_id': entry_id, 'name': name,
                'photo_url': f'/api/face-library/{entry_id}/photo', 'similarity': similarity}


class OneToOneResolutionTests(unittest.TestCase):
    """一对一消解：同一库内编号只给得分最高的脸，被抢的顺延到下一个候选，都没有就算未识别。"""

    @staticmethod
    def _cand(entry_id, name, similarity):
        return {'entry_id': entry_id, 'name': name,
                'photo_url': f'/api/face-library/{entry_id}/photo', 'similarity': similarity}

    def test_higher_similarity_wins_and_loser_falls_back(self):
        outcome = face_recognition.resolve_one_to_one({
            'F1': [self._cand('0001', '甲', .50), self._cand('0002', '乙', .45)],
            'F2': [self._cand('0001', '甲', .60)],
        }, threshold=.4)
        self.assertEqual(outcome['F2']['assigned']['entry_id'], '0001')
        self.assertEqual(outcome['F2']['assigned']['rank'], 1)
        self.assertEqual(outcome['F1']['assigned']['entry_id'], '0002')
        self.assertEqual(outcome['F1']['assigned']['rank'], 2)
        self.assertEqual(outcome['F1']['skipped'][0]['entry_id'], '0001')
        self.assertEqual(outcome['F1']['skipped'][0]['taken_by'], 'F2')

    def test_face_without_alternative_stays_unassigned(self):
        outcome = face_recognition.resolve_one_to_one({
            'F1': [self._cand('0001', '甲', .50)],
            'F2': [self._cand('0001', '甲', .60)],
        }, threshold=.4)
        self.assertIsNotNone(outcome['F2']['assigned'])
        self.assertIsNone(outcome['F1']['assigned'])
        self.assertEqual(outcome['F1']['skipped'][0]['taken_by'], 'F2')

    def test_fallback_only_counts_candidates_above_threshold(self):
        """备选低于阈值时不能顺延过去，直接算未识别。"""
        outcome = face_recognition.resolve_one_to_one({
            'F1': [self._cand('0001', '甲', .50), self._cand('0002', '乙', .30)],
            'F2': [self._cand('0001', '甲', .60)],
        }, threshold=.4)
        self.assertIsNone(outcome['F1']['assigned'])

    def test_recursive_fallback_to_third_candidate(self):
        outcome = face_recognition.resolve_one_to_one({
            'F1': [self._cand('0001', '甲', .50), self._cand('0002', '乙', .45), self._cand('0003', '丙', .42)],
            'F2': [self._cand('0001', '甲', .70)],
            'F3': [self._cand('0002', '乙', .65)],
        }, threshold=.4)
        self.assertEqual(outcome['F1']['assigned']['entry_id'], '0003')
        self.assertEqual(outcome['F1']['assigned']['rank'], 3)
        self.assertEqual([s['entry_id'] for s in outcome['F1']['skipped']], ['0001', '0002'])

    def test_same_entry_never_assigned_twice(self):
        outcome = face_recognition.resolve_one_to_one({
            'F1': [self._cand('0001', '甲', .80)],
            'F2': [self._cand('0001', '甲', .70)],
            'F3': [self._cand('0001', '甲', .60)],
        }, threshold=.4)
        assigned = [o['assigned']['entry_id'] for o in outcome.values() if o['assigned']]
        self.assertEqual(assigned, ['0001'])
        self.assertEqual(outcome['F1']['assigned']['similarity'], .80)
        self.assertEqual(len(outcome['F2']['skipped']) + len(outcome['F3']['skipped']), 2)


class QualityGateTests(unittest.TestCase):
    def test_confidence_gate_is_configurable(self):
        """远景小脸置信度天然低于 0.8，门槛必须可调，否则全部无法进入比对。"""
        import numpy as np

        from scrfd_detector import assess_quality
        face = {'bbox': [20, 20, 80, 90], 'confidence': .6,
                'landmarks': [[35, 40], [65, 40], [50, 55], [40, 75], [60, 75]]}
        image = np.random.default_rng(8).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        # 默认 0.8：0.6 置信度的人脸被拒
        self.assertIn('人脸置信度不足', assess_quality(image, face)[0])
        # 门槛降到检出阈值 0.5 后，置信度不再拦截
        self.assertNotIn('人脸置信度不足', assess_quality(image, face, min_confidence=.5)[0])
        # 置信度足够时默认门槛放行（保持匿名人脸流水线的原有行为）
        confident = {**face, 'confidence': .95}
        self.assertEqual(assess_quality(image, confident)[0], [])

    def test_size_and_sharpness_gates_are_configurable(self):
        import numpy as np

        from scrfd_detector import assess_quality
        face = {'bbox': [20, 20, 40, 40], 'confidence': .95,
                'landmarks': [[25, 26], [35, 26], [30, 31], [26, 35], [34, 35]]}
        image = np.random.default_rng(8).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        self.assertIn('原始人脸尺寸过小', assess_quality(image, face)[0])
        self.assertNotIn('原始人脸尺寸过小', assess_quality(image, face, min_size=5)[0])
        blank = np.zeros_like(image)
        self.assertIn('人脸模糊', assess_quality(blank, {**face, 'bbox': [20, 20, 80, 90]})[0])
        self.assertNotIn('人脸模糊', assess_quality(blank, {**face, 'bbox': [20, 20, 80, 90]}, min_sharpness=0)[0])


if __name__ == '__main__':
    unittest.main()
