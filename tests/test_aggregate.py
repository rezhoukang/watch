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
import head_fusion
import pose_roi
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


def head_detections() -> dict:
    """目标检测（沿用图片推理的目标检测）：medium 两个基准头 + RT-DETR 一个补齐头。"""
    base = {"detections": [{"class_id": 0, "class_name": 'head', "confidence": .9, "bbox": [10, 10, 60, 70]},
                           {"class_id": 0, "class_name": 'head', "confidence": .8, "bbox": [110, 10, 160, 70]}],
            "classes_used": ['head']}
    incoming = {"detections": [{"class_id": 0, "class_name": 'head', "confidence": .6, "bbox": [210, 10, 260, 70]}],
                "classes_used": ['head']}
    return head_fusion.fuse_results(base, incoming)


def pose_result() -> Results:
    """姿势检测：同样两个人框，带 17 点骨架。"""
    base = keypoints_of()
    return Results(np.full((400, 640, 3), 255, dtype=np.uint8), path='people.png', names={0: 'person'},
                   boxes=torch.tensor([[10, 10, 90, 200, .9, 0], [110, 10, 190, 200, .8, 0]],
                                      dtype=torch.float32),
                   keypoints=torch.tensor([base, keypoints_of(100)], dtype=torch.float32))


class BindFacesTests(unittest.TestCase):
    """姓名归属：只保留 IoU > 0 的配对，链式贪心 + 一一对应。"""

    def setUp(self):
        self.targets = [
            {"bbox": [0, 0, 100, 300], "class_name": 'head'},    # 大框
            {"bbox": [10, 10, 60, 120], "class_name": 'head'},   # 小框（与脸重叠更多）
            {"bbox": [400, 0, 500, 300], "class_name": 'head'},  # 远处
        ]

    def bind(self, bbox):
        faces = [{"face_id": 'F01', "bbox": bbox}]
        return aggregate.bind_faces(faces, self.targets).get('F01')

    def test_highest_overlap_wins(self):
        binding = self.bind([20, 20, 50, 60])
        assert binding is not None
        self.assertEqual(binding['target_index'], 1)
        self.assertAlmostEqual(binding['iou'], aggregate.iou([20, 20, 50, 60], [10, 10, 60, 120]), places=4)

    def test_face_without_overlap_is_not_bound(self):
        self.assertIsNone(self.bind([300, 100, 330, 140]))   # 离得近但不重叠 → 不硬塞
        self.assertIsNone(self.bind([1000, 900, 1040, 960]))

    def test_one_to_one_never_puts_two_faces_on_one_head(self):
        # 两张脸完全重合（同分同距，按传入顺序先到先得）而只有一个人头 → 只能贴一个
        targets = [{"bbox": [10, 10, 60, 120], "class_name": 'head'}]
        faces = [{"face_id": 'F01', "bbox": [20, 20, 50, 60]}, {"face_id": 'F02', "bbox": [20, 20, 50, 60]}]
        bound = aggregate.bind_faces(faces, targets)
        self.assertEqual(list(bound), ['F01'])
        self.assertEqual(bound['F01']['target_index'], 0)

    def test_two_faces_take_two_different_heads(self):
        bound = aggregate.bind_faces(
            [{"face_id": 'F01', "bbox": [20, 20, 50, 60]}, {"face_id": 'F02', "bbox": [22, 22, 52, 62]}],
            self.targets)
        self.assertEqual(len(bound), 2)
        self.assertEqual({item['target_index'] for item in bound.values()}, {0, 1})

    def test_no_targets_means_no_binding(self):
        self.assertEqual(aggregate.bind_faces([{"face_id": 'F01', "bbox": [1, 2, 3, 4]}], []), {})


class BindPosesTests(unittest.TestCase):
    """姿势绑定：只保留 IoU > 0 的配对，链式贪心 + 一一对应（IoU 并列时取中心距离近的）。"""

    def setUp(self):
        self.targets = [{"bbox": [0, 0, 100, 200]}, {"bbox": [300, 0, 400, 200]}]

    def test_person_box_containing_a_head_binds_by_iou(self):
        bound = aggregate.bind_poses([{"bbox": [5, 5, 95, 195]}], self.targets)
        self.assertEqual(bound[0]['target_index'], 0)
        self.assertGreater(bound[0]['iou'], 0)

    def test_pose_without_overlap_stays_unbound(self):
        self.assertEqual(aggregate.bind_poses([{"bbox": [1000, 1000, 1010, 1010]}], self.targets), {})

    def test_highest_iou_wins_even_if_not_the_first_target(self):
        targets = [{"bbox": [0, 0, 400, 400]}, {"bbox": [10, 10, 60, 120]}]
        bound = aggregate.bind_poses([{"bbox": [20, 20, 50, 60]}], targets)
        self.assertEqual(bound[0]['target_index'], 1)

    def test_one_to_one_never_reuses_a_head(self):
        targets = [{"bbox": [0, 0, 100, 200]}]
        bound = aggregate.bind_poses([{"bbox": [0, 0, 100, 200]}, {"bbox": [0, 0, 100, 50]}], targets)
        self.assertEqual(list(bound), [0], '只有 IoU 更大的那个绑上，人头不会被重复占用')
        self.assertEqual(bound[0]['iou'], 1.0)
        self.assertNotIn(1, bound)

    def test_equal_iou_falls_back_to_the_nearer_center(self):
        # 两个人头面积相同且都被姿势框包住 → IoU 相等，按中心距离近的优先
        targets = [{"bbox": [0, 0, 50, 50]}, {"bbox": [10, 10, 60, 60]}]
        pose = [{"bbox": [0, 0, 100, 100]}]
        self.assertAlmostEqual(aggregate.iou(pose[0]['bbox'], targets[0]['bbox']),
                               aggregate.iou(pose[0]['bbox'], targets[1]['bbox']), places=5)
        self.assertEqual(aggregate.bind_poses(pose, targets)[0]['target_index'], 1)

    def test_malformed_or_missing_boxes_are_skipped(self):
        self.assertEqual(aggregate.bind_poses([{"bbox": [1, 2, 3]}], self.targets), {})
        self.assertEqual(aggregate.bind_poses([{}], self.targets), {})


class RecognizeFacesTests(unittest.TestCase):
    """整图 1:N：保留人脸框供归属，并且一个库内编号只分配给一张脸。"""

    def setUp(self):
        self.faces = [
            {"bbox": [10, 10, 60, 80], "landmarks": [[20, 30]]*5, "confidence": .9},
            {"bbox": [100, 100, 130, 140], "landmarks": [[110, 110]]*5, "confidence": .8},
        ]
        detector = Mock()
        detector.detect.return_value = self.faces
        self.patch('_get_detector', Mock(return_value=detector))
        self.patch('assess_quality', Mock(return_value=([], {'sharpness': 100})))
        self.patch('extract_embedding', Mock(return_value=np.ones(512, dtype=np.float32)))
        self.patch('_load_entries', Mock(return_value={'0001': {'name': '张三'}}))
        self.patch('_load_embeddings', Mock(return_value=(['0001'], np.ones((1, 512), dtype=np.float32))))

    def patch(self, name: str, value) -> None:
        """打补丁并登记还原，避免 Mock 泄漏到同进程里的其他测试文件。"""
        mocked = patch.object(face_recognition, name, value)
        mocked.start()
        self.addCleanup(mocked.stop)

    def candidates(self, similarity: float = .62) -> None:
        self.patch('match_top_k', Mock(return_value=[{'entry_id': '0001', 'name': '张三', 'similarity': similarity}]))

    def test_big_face_wins_duplicate_identity_and_small_one_is_recorded_as_taken(self):
        self.candidates()
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
        self.candidates()
        result = face_recognition.recognize_faces(np.full((400, 400, 3), 255, dtype=np.uint8), threshold=.9)
        self.assertEqual([f['status'] for f in result['faces']], ['unknown', 'unknown'])
        self.assertEqual({f['match']['reason'] for f in result['faces']}, {'below_threshold'})

    def test_quality_gate_keeps_boxes_for_traceability(self):
        self.candidates()
        self.patch('assess_quality', Mock(return_value=(['原始人脸尺寸过小'], {'sharpness': 1})))
        result = face_recognition.recognize_faces(np.full((400, 400, 3), 255, dtype=np.uint8))
        self.assertEqual([f['status'] for f in result['faces']], ['rejected', 'rejected'])
        self.assertEqual(result['faces'][0]['bbox'], [10, 10, 60, 80])
        self.assertEqual(result['faces'][0]['reasons'], ['原始人脸尺寸过小'])
        self.assertIsNone(result['faces'][0]['match'])


class AggregateApiTests(unittest.TestCase):
    """HTTP 端到端：人头检测 → 人脸识别贴名 → 姿势绑定，后两步可分别跳过。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'uploads').mkdir()
        Image.new('RGB', (640, 400), 'white').save(self.root / 'uploads' / 'people.png')
        self.pose_model = Mock()
        self.pose_model.predict.return_value = [pose_result()]
        self.detection = Mock(return_value=head_detections())
        self.recognition = Mock(return_value=self.recognized())
        for module, field, value in (
            (common, 'UPLOAD_DIR', self.root / 'uploads'),
            (aggregate, 'RESULT_DIR', self.root / 'aggregate'),
            (common, '_require_head_detection', Mock()),
            (head_fusion, 'detect_image', self.detection),
            (face_recognition, 'list_entries', Mock(return_value=[{'id': '0001', 'name': '张三'}])),
            (face_recognition, 'recognize_faces', self.recognition),
        ):
            mocked = patch.object(module, field, value); mocked.start(); self.addCleanup(mocked.stop)
        model = patch.object(vision, 'get_model', lambda weights, classes=None: self.pose_model)
        model.start(); self.addCleanup(model.stop)
        self.client = TestClient(app.app)

    def patch(self, name: str, value) -> None:
        """打补丁并登记还原，避免 Mock 泄漏给同类里的后续用例。"""
        mocked = patch.object(face_recognition, name, value)
        mocked.start()
        self.addCleanup(mocked.stop)

    @staticmethod
    def recognized(bbox=None) -> dict:
        face = {"face_id": 'F01', "bbox": bbox or [30, 25, 50, 55], "confidence": .9, "face_pixels": 20,
                "landmarks": [[35, 33], [45, 33], [40, 41], [36, 48], [44, 48]], "status": 'matched',
                "reasons": [], "candidates": [], "photo_url": None,
                "match": {"matched": True, "entry_id": '0001', "name": '张三', "similarity": .62,
                          "photo_url": "/api/face-library/0001/photo", "rank": 1}}
        return {"status": 'ok', "detected": 1, "truncated": 0, "face_count": 1, "reassigned": 0,
                "matched": 1, "unknown": 0, "rejected": 0, "errors": 0, "threshold": .4,
                "detect_confidence": .5, "faces": [face]}

    def post(self, **overrides):
        return self.client.post('/api/aggregate/image/people.png', json=overrides)

    def test_target_detection_is_locked_to_image_inference_settings(self):
        record = self.post().json()
        # 前端不再传目标检测参数：后端固定 medium.pt 0.45 基准 + RT-DETRv4 0.25 补齐，IoU 规则锁死
        self.assertEqual(self.detection.call_args.kwargs,
                         {'rtdetr_base': False, 'merge_iou': .1, 'merge_confidence': .55})
        self.assertEqual(record['steps']['detect']['mode'], 'head_fusion')
        self.assertIn('合并 IoU ≤ 0.1', record['steps']['detect']['rule'])
        self.assertIn('自定义 IoU 生效置信度 > 0.55', record['steps']['detect']['rule'])

    def test_stale_target_detection_params_are_ignored(self):
        record = self.post(weights='yolov8m-pose.pt', confidence=.9, imgsz=1280,
                           classes='book').json()
        self.assertEqual(self.detection.call_args.kwargs['merge_iou'], .1)
        self.assertEqual(record['count'], 3)

    def test_names_are_attached_to_the_head_and_poses_are_bound(self):
        response = self.post()
        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()
        self.assertEqual(record['count'], 3)
        # 人脸中心 (40, 40) 落在第一个人头框内 → 姓名贴到它，其他人头不受影响
        self.assertEqual(record['targets'][0]['name'], '张三')
        self.assertEqual(record['targets'][0]['similarity'], .62)
        self.assertEqual(record['targets'][0]['face_id'], 'F01')
        self.assertNotIn('name', record['targets'][1])
        self.assertEqual(record['faces'][0]['target_index'], 0)
        self.assertEqual(record['summary']['named_targets'], 1)
        self.assertEqual(record['summary']['bound_faces'], 1)
        # 两个人头中心分别落在各自的 Pose 人体框内 → 默认策略（contain）全部绑定
        self.assertEqual([pose['target_index'] for pose in record['poses']], [0, 1])
        self.assertEqual(record['steps']['pose']['bound'], 2)
        self.assertEqual(record['summary']['bound_poses'], 2)
        self.assertEqual(record['targets'][0]['poses'][0]['bind']['iou'], .1974)
        self.assertIn(record['targets'][0]['poses'][0]['behavior']['state'], ('低头', '抬头', '无法判断'))
        # 三步都要有各自的统计与耗时，便于界面解释
        self.assertEqual(record['steps']['detect']['count'], 3)
        self.assertEqual(record['steps']['face']['threshold'], aggregate.MATCH_THRESHOLD)
        self.assertEqual(record['steps']['face']['library_count'], 1)
        self.assertEqual(record['steps']['pose']['weight'], 'yolov8m-pose.pt')
        for step in ('detect', 'face', 'pose'):
            self.assertIn('elapsed_seconds', record['steps'][step], '每步都要报耗时，界面才好解释')
        # 结果图要落地且能通过接口取回
        self.assertTrue((aggregate.RESULT_DIR / record['result']).is_file())
        image = self.client.get(record['result_url'])
        self.assertEqual(image.status_code, 200, image.text)
        self.assertEqual(image.headers['content-type'], 'image/png')

    def test_unchecked_steps_are_skipped_and_reported(self):
        record = self.post(face=False, pose=False).json()
        self.assertEqual(record['count'], 3)
        self.assertFalse(record['steps']['face']['enabled'])
        self.assertFalse(record['steps']['pose']['enabled'])
        self.assertEqual(record['faces'], [])
        self.assertEqual(record['poses'], [])
        self.assertTrue(all('name' not in target for target in record['targets']))
        # 跳过姿势检测时不该再加载 Pose 权重，也不该去读人脸库
        self.assertEqual(self.pose_model.predict.call_count, 0)
        self.assertEqual(self.recognition.call_count, 0)

    def test_pose_keypoints_come_from_the_pose_model(self):
        record = self.post().json()
        self.assertEqual(self.pose_model.predict.call_count, 1)
        self.assertEqual(self.pose_model.predict.call_args.kwargs['conf'], .6)
        self.assertEqual(len(record['poses'][0]['keypoints']), 17)

    def test_pose_settings_are_locked_and_stale_overrides_are_ignored(self):
        with patch.object(vision, 'detect_image', wraps=vision.detect_image) as detect:
            response = self.post(face=False, pose_confidence=.01, pose_weights='other.pt',
                                 imgsz=1280, head_down_r_threshold=.5,
                                 head_up_r_threshold=.6, pose_roi=False, roi_confidence=.9)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(detect.call_args.kwargs, {
            'weights': 'yolov8m-pose.pt', 'conf': .6, 'imgsz': 640, 'save_result': False,
            'head_down_r_threshold': .05, 'head_up_r_threshold': .20,
            'pose_roi': True, 'roi_confidence': .03,
        })
        self.assertEqual(self.pose_model.predict.call_args.kwargs['conf'], .6)
        self.assertEqual(self.pose_model.predict.call_args.kwargs['imgsz'], 640)
        step = response.json()['steps']['pose']
        for key, value in {'weight': 'yolov8m-pose.pt', 'confidence': .6, 'imgsz': 640,
                           'head_down_r_threshold': .05, 'head_up_r_threshold': .20,
                           'pose_roi': True, 'roi_confidence': .03}.items():
            self.assertEqual(step[key], value)
        self.recognition.assert_not_called()

    def test_unknown_poses_are_excluded_and_empty_results_are_valid(self):
        result = pose_result()
        assert result.keypoints is not None
        # R=0.1 落在两个阈值中间；另一个人的鼻关键点缺失。
        result.keypoints.data[0, 0, 1] = 35
        result.keypoints.data[1, 0, 2] = 0
        self.pose_model.predict.return_value = [result]
        with patch.object(pose_roi, '_get_model') as objects:
            response = self.post(face=False)
        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()
        self.assertEqual(record['poses'], [])
        self.assertEqual(record['steps']['pose']['excluded_unknown'], 2)
        self.assertEqual(record['steps']['pose']['bound'], 0)
        self.assertEqual(record['count'], 3)
        objects.assert_not_called()

    def test_head_down_objects_use_full_image_association_at_point_zero_three(self):
        result = pose_result()
        assert result.keypoints is not None
        result.keypoints.data[:, 0, 1] = 42  # R=-0.04，低头。
        self.pose_model.predict.return_value = [result]
        object_model = Mock()
        object_model.predict.return_value = [Results(
            result.orig_img, path='people.png', names={0: 'cell phone', 1: 'book'},
            boxes=torch.tensor([[20, 50, 35, 65, .04, 0], [120, 50, 140, 70, .04, 1]],
                               dtype=torch.float32))]
        with patch.object(pose_roi, '_get_model', return_value=object_model):
            response = self.post(face=False)
        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()
        self.assertEqual([p['behavior']['state'] for p in record['poses']], ['疑似看手机', '疑似阅读'])
        self.assertEqual([p['target_index'] for p in record['poses']], [0, 1])
        self.assertEqual(object_model.predict.call_count, 1)
        self.assertEqual(object_model.predict.call_args.kwargs['conf'], .03)
        self.assertEqual(object_model.predict.call_args.kwargs['imgsz'], 640)
        self.assertEqual(object_model.predict.call_args.kwargs['source'].shape, (400, 640, 3))
        self.assertEqual(record['targets'][0]['poses'][0]['behavior']['state'], '疑似看手机')

    def test_unbound_pose_is_listed_instead_of_silently_dropped(self):
        far = Results(np.full((400, 640, 3), 255, dtype=np.uint8), path='people.png', names={0: 'person'},
                      boxes=torch.tensor([[600, 380, 639, 399, .9, 0]], dtype=torch.float32),
                      keypoints=torch.tensor([keypoints_of()], dtype=torch.float32))
        self.pose_model.predict.return_value = [far]
        record = self.post().json()
        self.assertEqual(record['unbound_poses'], [1])
        self.assertEqual(record['steps']['pose']['bound'], 0)

    def test_face_outside_every_head_is_reported_as_unbound(self):
        self.patch('recognize_faces', Mock(return_value=self.recognized(bbox=[600, 380, 630, 399])))
        record = self.post().json()
        self.assertEqual(record['unbound_faces'], ['F01'])
        self.assertIsNone(record['faces'][0]['target_index'])
        self.assertEqual(record['summary']['named_targets'], 0)

    def test_binding_rule_is_fixed_and_reported(self):
        record = self.post().json()
        self.assertEqual(record['steps']['pose']['min_iou'], 0)
        self.assertIn('链式贪心', record['bind']['rule'])
        self.assertIn('IoU > 0', record['bind']['rule'])
        self.assertIn('链式贪心', record['bind']['face_rule'])
        # 人头框与人体框的 IoU 很小（≈0.2），但只要 > 0 就该绑上
        self.assertLess(record['steps']['pose']['bound'] and record['targets'][0]['poses'][0]['bind']['iou'], .3)

    def test_face_parameters_are_locked(self):
        """人脸识别参数全部锁死：界面只有一个勾选框，阈值由后端常量决定。"""
        record = self.post().json()
        self.assertEqual(record['steps']['face']['threshold'], aggregate.MATCH_THRESHOLD)
        self.assertEqual(record['steps']['face']['detect_confidence'], aggregate.FACE_CONFIDENCE)
        called = self.recognition.call_args.kwargs
        self.assertEqual(called['threshold'], aggregate.MATCH_THRESHOLD)
        self.assertEqual(called['detect_confidence'], aggregate.FACE_CONFIDENCE)
        self.assertEqual(called['min_confidence'], aggregate.MIN_QUALITY_CONFIDENCE)
        self.assertEqual(called['min_size'], aggregate.MIN_FACE_PIXELS)
        self.assertEqual(called['min_sharpness'], aggregate.MIN_FACE_SHARPNESS)
        # 锁定的值就是之前那个下限：人脸置信度 ≥ 0.5、识别阈值 ≥ 0.3
        self.assertEqual((aggregate.FACE_CONFIDENCE, aggregate.MATCH_THRESHOLD), (.5, .3))
        # 前端即使多传阈值也被忽略，不会改变生效值
        ignored = self.post(threshold=.9, face_confidence=.9).json()
        self.assertEqual(ignored['steps']['face']['threshold'], aggregate.MATCH_THRESHOLD)
        self.assertEqual(ignored['steps']['face']['detect_confidence'], aggregate.FACE_CONFIDENCE)

    def test_empty_library_is_reported(self):
        # 直接赋值会泄漏给同类里的后续用例（后面几个都要求人脸识别可用），必须登记还原
        self.patch('list_entries', Mock(return_value=[]))
        response = self.post()
        self.assertEqual(response.status_code, 400)
        self.assertIn('人脸库为空', response.json()['detail'])
        self.assertEqual(self.post(face=False).status_code, 200)

    def test_unknown_extra_fields_are_ignored(self):
        # 绑定规则已固定：前端误传 bind / bind_iou 不再是一个开关，只是被忽略
        record = self.post(bind='bogus', bind_iou=1.5, weights='yolov8m.pt').json()
        self.assertIn('链式贪心', record['bind']['rule'])
        self.assertEqual(record['steps']['pose']['min_iou'], 0)

    def test_out_of_range_or_unknown_fields_do_not_break_the_run(self):
        # 阈值参数已锁死：多传/乱传一律忽略，不会 422，也不会改变生效值
        extra = self.post(threshold=2, face_confidence=3)
        self.assertEqual(extra.status_code, 200, extra.text)
        self.assertEqual(extra.json()['steps']['face']['threshold'], aggregate.MATCH_THRESHOLD)

    def test_missing_and_illegal_filenames_are_rejected(self):
        self.assertEqual(self.client.post('/api/aggregate/image/none.png', json={}).status_code, 404)
        self.assertEqual(self.client.post('/api/aggregate/image/a..b.png', json={}).status_code, 400)


if __name__ == '__main__':
    unittest.main()
