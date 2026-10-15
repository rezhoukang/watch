import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import pose_roi


def detection(state='低头', dx=0):
    points: list[list[float]] = [[0, 0, 0] for _ in range(17)]
    for i, xy in zip(range(5, 11), [(40, 40), (160, 40), (30, 110), (170, 110), (70, 150), (130, 150)]):
        points[i] = [xy[0]+dx, xy[1], .9]
    return {'bbox': [30+dx, 40, 170+dx, 150], 'keypoints': points,
            'behavior': {'state': state, 'head': {'evidence': 'preserved'}}}


def prediction(*items):
    return SimpleNamespace(boxes=[SimpleNamespace(cls=[cls], conf=[confidence],
                           xyxy=[np.array(box)])
                           for cls, confidence, box in items])


class PoseRoiTests(unittest.TestCase):
    def run_refine(self, detections, *predictions, enabled=True):
        model = Mock()
        model.predict.return_value = list(predictions)
        with patch.object(pose_roi, '_get_model', return_value=model):
            pose_roi.refine(np.zeros((200, 600, 3), dtype=np.uint8), detections, enabled)
        return model

    def test_phone_priority_reading_and_original_coordinates(self):
        people = [detection(), detection(dx=200), detection(dx=400)]
        model = self.run_refine(people,
                        prediction((0, .85, [65, 115, 85, 135]), (1, .9, [90, 90, 130, 120]),
                                   (1, .75, [300, 60, 340, 80])))
        model.predict.assert_called_once()
        self.assertEqual(model.predict.call_args.kwargs['source'].shape, (200, 600, 3))
        self.assertEqual([d['behavior']['state'] for d in people], ['疑似看手机', '疑似阅读', '低头'])
        obj = next(o for o in people[0]['behavior']['roi']['objects'] if o['kind'] == 'phone')
        self.assertEqual(obj['bbox'], [65, 115, 85, 135])
        self.assertEqual(people[0]['behavior']['head'], {'evidence': 'preserved'})

    def test_low_confidence_and_outside_retain_head_down_missing_joints_allow_binding(self):
        people = [detection()]
        self.run_refine(people, prediction((0, .029, [65, 115, 85, 135]),
                                          (2, .99, [65, 115, 85, 135]),
                                          (0, .95, [0, 140, 20, 150]),
                                          (1, .99, [210, 0, 250, 200])))
        self.assertEqual(people[0]['behavior']['state'], '低头')
        self.assertEqual(people[0]['behavior']['roi']['objects'], [])
        for i in range(5, 11):
            person = detection()
            person['keypoints'][i][2] = .59
            self.run_refine([person], prediction((0, .9, [65, 115, 85, 135])))
            self.assertEqual(person['behavior']['state'], '疑似看手机')
            self.assertEqual(person['behavior']['roi']['polygon'], [[30, 40], [170, 40], [170, 150], [30, 150]])

    def test_only_head_down_processed_and_shared_objects_assigned_once(self):
        unchanged = [detection('举手'), detection('抬头'), detection('无法判断')]
        before = copy.deepcopy(unchanged)
        self.run_refine(unchanged).predict.assert_not_called()
        self.assertEqual(unchanged, before)
        people = [detection(), detection()]
        phone = (0, .8, [65, 115, 85, 135])
        self.run_refine(people, prediction(phone))
        self.assertEqual(sum(d['behavior']['state'] == '疑似看手机' for d in people), 1)

    def test_maximum_bbox_intersection_beats_keypoint_hull_and_iou(self):
        people = [detection(), detection()]
        rectangles = [[(10, 10), (180, 10), (10, 100), (180, 100), (10, 180), (180, 180)],
                      [(100, 40), (130, 40), (100, 60), (130, 60), (110, 80), (120, 80)]]
        for person, coordinates in zip(people, rectangles):
            for i, xy in zip(range(5, 11), coordinates):
                person['keypoints'][i] = [*xy, .9]
        # Deliberately reverse the keypoint hull areas; only Pose boxes count.
        people[0]['bbox'] = [10, 10, 180, 180]
        people[1]['bbox'] = [100, 40, 130, 80]
        people[0]['keypoints'], people[1]['keypoints'] = people[1]['keypoints'], people[0]['keypoints']
        self.run_refine(people, prediction((0, .9, [90, 40, 130, 80])))
        self.assertEqual([d['behavior']['state'] for d in people], ['疑似看手机', '低头'])
        self.assertEqual(people[0]['behavior']['roi']['objects'][0]['intersection_area'], 1600)

    def test_partial_overlap_with_center_outside_is_allowed(self):
        people = [detection()]
        self.run_refine(people, prediction((0, .9, [0, 100, 50, 140])))
        self.assertEqual(people[0]['behavior']['state'], '疑似看手机')
        self.assertGreater(people[0]['behavior']['roi']['objects'][0]['intersection_area'], 0)

    def test_invalid_bbox_retains_head_down_without_running_model(self):
        for box in ([], [10, 10, 10, 20], [0, float('nan'), 20, 30], [700, 10, 800, 20]):
            person = detection()
            person['bbox'] = box
            self.run_refine([person]).predict.assert_not_called()
            self.assertEqual(person['behavior']['state'], '低头')
            self.assertEqual(person['behavior']['roi']['status'], 'invalid_pose_bbox')

    def test_disabled_or_unavailable_retains_head_down(self):
        person = detection()
        self.run_refine([person], enabled=False).predict.assert_not_called()
        self.assertEqual(person['behavior']['roi']['status'], 'disabled')
        with patch.object(pose_roi, '_get_model', side_effect=RuntimeError('no model')):
            pose_roi.refine(np.zeros((200, 200, 3), dtype=np.uint8), [person])
        self.assertEqual(person['behavior']['state'], '低头')
        self.assertEqual(person['behavior']['roi']['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
