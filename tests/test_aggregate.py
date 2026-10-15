"""聚合操作：绑定策略（纯函数）、整图 1:N 归属、以及 HTTP 端到端。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image
from ultralytics.engine.results import Results

import aggregate
import app
import face_recognition
import vision
from web import common


def keypoints_of(offset: float = 0.0) -> list:
    """一份可用的 COCO 17 点骨架（肩宽足够，行为判定不会退化）。"""
    points = [[0.0, 0.0, 0.0] for _ in range(17)]
    points[0] = [55+offset, 29, .9]
    points[5], points[6] = [30+offset, 40, .9], [80+offset, 40, .9]
    points[7], points[8] = [20+offset, 60, .9], [90+offset, 60, .9]
    points[9], points[10] = [25+offset, 80, .9], [85+offset, 80, .9]
    points[11], points[12] = [35+offset, 70, .9], [75+offset, 70, .9]
    return points


def target_result() -> Results:
    """目标检测：两个人框，不带骨架。"""
    return Results(np.full((400, 640, 3), 255, dtype=np.uint8), path='people.png', names={0: 'person'},
                   boxes=torch.tensor([[10, 10, 90, 200, .9, 0], [110, 10, 190, 200, .8, 0]],
                                      dtype=torch.float32))


def pose_result() -> Results:
    """姿势检测：同样两个人框，带 17 点骨架。"""
    base = keypoints_of()
    return Results(np.full((400, 640, 3), 255, dtype=np.uint8), path='people.png', names={0: 'person'},
                   boxes=torch.tensor([[10, 10, 90, 200, .9, 0], [110, 10, 190, 200, .8, 0]],
                                      dtype=torch.float32),
                   keypoints=torch.tensor([[base, keypoints_of(100)]], dtype=torch.float32))


class BindFacesTests(unittest.TestCase):
    """姓名归属：先看谁包住人脸中心，都不包含时退化为最近目标中心。"""

    def setUp(self):
        self.targets = [
            {"bbox": [0, 0, 100, 300], "class_name": 'person'},    # 大框
            {"bbox": [10, 10, 60, 120], "class_name": 'phone'},    # 小框，更贴合
            {"bbox": [400, 0, 500, 300], "class_name": 'person'},  # 远处
        ]

    def bind(self, bbox):
        faces = [{"face_id": 'F01', "bbox": bbox}]
        return aggregate.bind_faces(faces, self.targets).get('F01')

    def test_smallest_containing_target_wins(self):
        binding = self.bind([20, 20, 50, 60])
        self.assertEqual(binding['target_index'], 1)
        self.assertEqual(binding['strategy'], 'center_inside')
        self.assertEqual(binding['iou'], 100*100/(40*150*2-100*100+100*100*0))  # 占位：见下方断言
        self.assertAlmostEqual(binding['iou'], aggregate.iou([20, 20, 50, 60], [10, 10, 60, 120]), places=4)

    def test_face_outside_all_targets_snaps_to_nearest_center(self):
        binding = self.bind([300, 100, 330, 140])
        self.assertEqual(binding['target_index'], 2)
        self.assertEqual(binding['strategy'], 'nearest_center')
        self.assertGreater(binding['distance'], 0)

    def test_far_away_face_is_not_bound(self):
        self.assertIsNone(self.bind([1000, 900, 1040, 960]))

    def test_no_targets_means_no_binding(self):
        self.assertEqual(aggregate.bind_faces([{"face_id": 'F01', "bbox": [1, 2, 3, 4]}], []), {})


class BindPosesTests(unittest.TestCase):
    """姿势绑定：三种策略都必须能解释“为什么绑给这个目标、为什么没绑”。"""

    def setUp(self):
        self.targets = [{"bbox": [0, 0, 100, 200]}, {"bbox": [300, 0, 400, 200]}]
        self.poses = [{"bbox": [5, 5, 95, 195]}, {"bbox": [1000, 1000, 1010, 1010]}]

    def test_iou_strategy_needs_threshold(self):
        bound = aggregate.bind_poses(self.poses, self.targets, 'iou', .5)
        self.assertEqual(bound[0]['target_index'], 0)
        self.assertGreaterEqual(bound[0]['iou'], .5)
        self.assertNotIn(1, bound)  # 离得远 → 未绑定，而不是硬塞给某个目标

    def test_iou_threshold_can_reject_everything(self):
        self.assertEqual(aggregate.bind_poses(self.poses, self.targets, 'iou', .99), {})

    def test_contain_strategy_uses_smallest_containing_box(self):
        targets = [{"bbox": [0, 0, 400, 400]}, {"bbox": [10, 10, 60, 120]}]
        bound = aggregate.bind_poses([{"bbox": [20, 20, 50, 60]}], targets, 'contain')
        self.assertEqual(bound[0]['target_index'], 1)

    def test_contain_strategy_skips_center_outside(self):
        self.assertEqual(aggregate.bind_poses([{"bbox": [500, 500, 520, 520]}], self.targets, 'contain'), {})

    def test_nearest_strategy_always_binds_when_targets_exist(self):
        bound = aggregate.bind_poses(self.poses, self.targets, 'nearest')
        self.assertEqual(bound[0]['target_index'], 0)
        self.assertEqual(bound[1]['target_index'], 1)
        self.assertGreater(bound[1]['distance'], bound[0]['distance'])

    def test_unknown_strategy_and_bad_threshold_are_rejected(self):
        with self.assertRaises(ValueError):
            aggregate.bind_poses(self.poses, self.targets, 'bogus')
        with self.assertRaises(ValueError):
            aggregate.bind_poses(self.poses, self.targets, 'iou', 1.5)


class RecognizeFacesTests(unittest.TestCase):
    """整图 1:N：保留人脸框供归属，并且一个库内编号只分配给一张脸。"""

    def setUp(self):
        self.faces = [
            {"bbox": [10, 10, 60, 80], "landmarks": [[20, 30]]*5, "confidence": .9},
            {"bbox": [100, 100, 130, 140], "landmarks": [[110, 110]]*5, "confidence": .8},
        ]
        detector = Mock()
        detector.detect.return_value = self.faces
        for module, field, value in (
            (face_recognition, '_get_detector', lambda: detector),
            (face_recognition, 'assess_quality', Mock(return_value=([], {'sharpness': 100}))),
            (face_recognition, 'extract_embedding', Mock(return_value=np.ones(512, dtype=np.float32))),
            (face_recognition, 'list_entries', Mock(return_value=[{'id': '0001', 'name': '张三'}])),
            (face_recognition, '_load_entries', Mock(return_value={'0001': {'name': '张三'}})),
            (face_recognition, '_load_embeddings', Mock(return_value((['0001'], np.ones((1, 512), dtype=np.float32)))),
        ):
            mocked = patch.object(module, field, Mock(side_effect=value) if callable(value) and field == '_get_detector' else value)
            mocked.start(); self.addCleanup(mocked.stop)

    def test_big_face_wins_duplicate_identity_and_small_one_is_recorded_as_taken(self):
        face_recognition.match_top_k = Mock(return_value=[{'entry_id': '0001', 'name': '张三', 'similarity': .62}])
        result = face_recognition.recognize_faces(np.full((400, 400, 3), 255, dtype=np.uint8), threshold=.4)
        self.assertEqual([f['face_id'] for f in result['faces']], ['F01', 'F02'])
        self.assertEqual(result['faces'][0]['status'], 'matched')
        self.assertEqual(result['faces'][0]['match']['name'], '张三')
        self.assertEqual(result['faces'][1]['status'], 'unknown')
        self.assertEqual(result['faces'][1]['match']['reason'], 'taken')
        self.assertEqual(result['faces'][1]['conflict']['skipped'][0]['taken_by'], 'F01')
        self.assertEqual(result['matched'], 1)
        self.assertEqual(result['reassigned'], 0)

    def test_below_threshold_is_distinguishable_from_taken(self):
        face_recognition.match_top_k = Mock(return_value=[{'entry_id': '0001', 'name': '张三', 'similarity': .62}])
        result = face_recognition.recognize_faces(np.full((400, 400, 3), 255, dtype=np.uint8), threshold=.9)
        self.assertEqual(result['faces'][0]['status'], 'unknown')
        self.assertEqual(result['faces'][0]['match']['reason'], 'below_threshold')
        self.assertEqual(result['faces'][1]['match']['reason'], 'below_threshold')

    def test_quality_gate_keeps_boxes_for_traceability(self):
        face_recognition.assess_quality = Mock(return_value=(['原始人脸尺寸过小'], {'sharpness': 1}))
        result = face_recognition.recognize_faces(np.full((400, 400, 3), 255, dtype=np.uint8))
        self.assertEqual([f['status'] for f in result['faces']], ['rejected', 'rejected'])
        self.assertEqual(result['faces'][0]['bbox'], [10, 10, 60, 80])
        self.assertEqual(result['faces'][0]['reasons'], ['原始人脸尺寸过小'])
        self.assertIsNone(result['faces'][0]['match'])


class AggregateApiTests(unittest.TestCase):
    """HTTP 端到端：目标检测 → 人脸识别贴名 → 姿势绑定，两步可分别跳过。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'uploads').mkdir()
        Image.new('RGB', (640, 400), 'white').save(self.root / 'uploads' / 'people.png')
        self.targets_model, self.pose_model = Mock(), Mock()
        self.targets_model.predict.return_value = [target_result()]
        self.pose_model.predict.return_value = [pose_result()]
        for module, field, value in (
            (common, 'UPLOAD_DIR', self.root / 'uploads'),
            (aggregate, 'RESULT_DIR', self.root / 'aggregate'),
            (common, '_require_image_backend', Mock()),
            (face_recognition, 'list_entries', Mock(return_value=[{'id': '0001', 'name': '张三'}])),
            (face_recognition, 'recognize_faces', Mock(return_value=self.recognized())),
        ):
            mocked = patch.object(module, field, value); mocked.start(); self.addCleanup(mocked.stop)
        model = patch.object(vision, 'get_model',
                             lambda weights, classes=None: self.pose_model if 'pose' in weights else self.targets_model)
        model.start(); self.addCleanup(model.stop)
        self.client = TestClient(app.app)

    @staticmethod
    def recognized() -> dict:
        face = {"face_id": 'F01', "bbox": [30, 20, 60, 55], "confidence": .9, "face_pixels": 30,
                "landmarks": [[35, 30], [50, 30], [42, 40], [36, 48], [52, 48]], "status": 'matched',
                "reasons": [], "candidates": [], "photo_url": None,
                "match": {"matched": True, "entry_id": '0001', "name": '张三', "similarity": .62,
                          "photo_url": "/api/face-library/0001/photo", "rank": 1}}
        return {"status": 'ok', "detected": 1, "truncated": 0, "face_count": 1, "reassigned": 0,
                "matched": 1, "unknown": 0, "rejected": 0, "errors": 0, "threshold": .4,
                "detect_confidence": .5, "faces": [face]}

    def post(self, **overrides):
        payload = {"weights": 'medium.pt', "confidence": .38, "imgsz": 640}
        payload.update(overrides)
        return self.client.post('/api/aggregate/image/people.png', json=payload)

    def test_names_are_attached_to_the_target_and_poses_are_bound(self):
        response = self.post()
        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()
        self.assertEqual(record['count'], 2)
        # 人脸中心 (45, 37.5) 落在第一个目标框内 → 姓名贴到它，第二个目标不受影响
        self.assertEqual(record['targets'][0]['name'], '张三')
        self.assertEqual(record['targets'][0]['similarity'], .62)
        self.assertEqual(record['targets'][0]['face_id'], 'F01')
        self.assertNotIn('name', record['targets'][1])
        self.assertEqual(record['faces'][0]['target_index'], 0)
        self.assertEqual(record['summary']['named_targets'], 1)
        # 两个姿势框都套在对应的目标框上 → 全部绑定，并记录绑定依据
        self.assertEqual([pose['target_index'] for pose in record['poses']], [0, 1])
        self.assertEqual(record['steps']['pose']['bound'], 2)
        self.assertEqual(record['summary']['bound_poses'], 2)
        self.assertEqual(record['targets'][0]['poses'][0]['bind']['strategy'], 'iou')
        self.assertIn('低头', (record['targets'][0]['poses'][0]['behavior']['state'],
                              record['targets'][0]['poses'][0]['behavior']['state']))
        # 三步都要有各自的配置与耗时，便于界面解释
        self.assertEqual(record['steps']['detect']['weight'], 'medium.pt')
        self.assertEqual(record['steps']['face']['threshold'], .4)
        self.assertEqual(record['steps']['face']['library_count'], 1)
        self.assertGreater(record['steps']['detect']['elapsed_seconds'], 0)
        # 结果图要落地且能通过接口取回
        self.assertTrue((aggregate.RESULT_DIR / record['result']).is_file())
        image = self.client.get(record['result_url'])
        self.assertEqual(image.status_code, 200, image.text)
        self.assertEqual(image.headers['content-type'], 'image/png')

    def test_unchecked_steps_are_skipped_and_reported(self):
        record = self.post(face=False, pose=False).json()
        self.assertEqual(record['count'], 2)
        self.assertFalse(record['steps']['face']['enabled'])
        self.assertFalse(record['steps']['pose']['enabled'])
        self.assertEqual(record['faces'], [])
        self.assertEqual(record['poses'], [])
        self.assertTrue(all('name' not in target for target in record['targets']))
        # 跳过姿势检测时不该再加载 Pose 权重
        self.assertEqual(self.pose_model.predict.call_count, 0)

    def test_pose_keypoints_are_reused_when_detection_weight_is_a_pose_model(self):
        record = self.post(weights='yolov8m-pose.pt', confidence=None).json()
        self.assertEqual(record['steps']['pose']['keypoints_source'], 'detect_weight')
        self.assertEqual(record['steps']['pose']['weight'], 'yolov8m-pose.pt')
        self.assertEqual(self.pose_model.predict.call_count, 0)
        self.assertEqual(self.targets_model.predict.call_args.kwargs['conf'], .6)  # Pose 权重默认置信度
        self.assertEqual(record['poses'][0]['source'], 'detect_weight')

    def test_pose_model_runs_when_detection_weight_has_no_keypoints(self):
        record = self.post().json()
        self.assertEqual(record['steps']['pose']['keypoints_source'], 'pose_model')
        self.assertEqual(self.pose_model.predict.call_count, 1)
        self.assertEqual(record['steps']['pose']['weight'], 'yolov8m-pose.pt')

    def test_unbound_pose_is_listed_instead_of_silently_dropped(self):
        far = Results(np.full((400, 640, 3), 255, dtype=np.uint8), path='people.png', names={0: 'person'},
                      boxes=torch.tensor([[600, 380, 639, 399, .9, 0]], dtype=torch.float32),
                      keypoints=torch.tensor([[keypoints_of()]], dtype=torch.float32))
        self.pose_model.predict.return_value = [far]
        record = self.post().json()
        self.assertEqual(record['unbound_poses'], [1])
        self.assertEqual(record['steps']['pose']['bound'], 0)

    def test_empty_library_is_reported(self):
        face_recognition.list_entries = Mock(return_value=[])
        response = self.post()
        self.assertEqual(response.status_code, 400)
        self.assertIn('人脸库为空', response.json()['detail'])

    def test_empty_library_is_fine_when_face_step_is_skipped(self):
        face_recognition.list_entries = Mock(return_value=[])
        self.assertEqual(self.post(face=False).status_code, 200)

    def test_unknown_bind_strategy_is_rejected(self):
        response = self.post(bind='bogus')
        self.assertEqual(response.status_code, 400)
        self.assertIn('未知的绑定策略', response.json()['detail'])

    def test_out_of_range_parameters_are_rejected(self):
        for bad in ({'bind_iou': 1.5}, {'threshold': 2}, {'imgsz': 100}, {'pose_confidence': 0}):
            response = self.post(**bad)
            self.assertEqual(response.status_code, 422, f'{bad} 应被拒绝：{response.text}')

    def test_missing_image_returns_404(self):
        self.assertEqual(self.client.post('/api/aggregate/image/none.png', json={}).status_code, 404)


if __name__ == '__main__':
    unittest.main()
