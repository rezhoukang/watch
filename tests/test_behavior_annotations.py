import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import app
import behavior_annotations as annotations
import dataset_store
import inference_history as history
import pose_behaviors
import pose_support
import vision


def points(raised=True):
    data = [[0.0, 0.0, 0.0] for _ in range(17)]
    data[0] = [100, 60, 0.9]
    data[5], data[6] = [80, 100, 0.9], [120, 100, 0.9]
    data[7], data[9] = [80, 70, 0.9], [80, 40 if raised else 90, 0.9]
    data[8], data[10] = [120, 130, 0.9], [120, 160, 0.9]
    return data


class HandRaiseTests(unittest.TestCase):
    def test_raise_side_priority_face_touch_and_missing_points(self):
        left = pose_behaviors.classify(points(), 200, 200)
        self.assertEqual(left["state"], "举手")
        self.assertEqual(left["hand_raise"]["side"], "left")
        both = points()
        both[8], both[10] = [120, 70, 0.9], [120, 40, 0.9]
        self.assertEqual(pose_behaviors.hand_raise(both, 200, 200)["side"], "both")
        face_touch = points()
        face_touch[9][1] = 55
        self.assertFalse(pose_behaviors.hand_raise(face_touch, 200, 200)["raised"])
        for index in (0, 5, 7, 9):
            sample = points()
            sample[index][2] = 0.59
            self.assertNotEqual(pose_behaviors.classify(sample, 200, 200)["state"], "举手")
        self.assertEqual(pose_behaviors.classify(points(False), 200, 200)["state"], "抬头")

    def test_exact_keypoint_threshold_and_eye_fallback(self):
        sample = points()
        sample[9][2] = 0.6
        self.assertTrue(pose_behaviors.hand_raise(sample, 200, 200)["raised"])
        sample[0] = [0, 0, 0]
        sample[1], sample[2] = [95, 60, 0.8], [105, 60, 0.8]
        self.assertEqual(pose_behaviors.hand_raise(sample, 200, 200)["head_source"], "eyes")

    def test_head_states_threshold_reliability_and_raise_priority(self):
        sample = points(False)
        for y, state in ((98.01, "低头"), (98, "无法判断"), (94.8, "无法判断"),
                         (92, "无法判断"), (91.99, "抬头"), (110, "低头")):
            sample[0] = [100, y, 0.6]
            self.assertEqual(pose_behaviors.classify(sample, 200, 200)["state"], state)
        for index in (0, 5, 6):
            bad = copy.deepcopy(sample)
            bad[index][2] = 0.59
            self.assertEqual(pose_behaviors.classify(bad, 200, 200)["state"], "无法判断")
        for bad_point in ([0, 60, .9], [100, float('nan'), .9], [100, 200, .9]):
            bad = copy.deepcopy(sample)
            bad[0] = bad_point
            self.assertEqual(pose_behaviors.classify(bad, 200, 200)["state"], "无法判断")
        self.assertEqual(pose_behaviors.classify([], 200, 200)["state"], "无法判断")
        sample[6] = sample[5].copy()
        self.assertEqual(pose_behaviors.classify(sample, 200, 200)["state"], "无法判断")
        raised = pose_behaviors.classify(points(), 200, 200)
        self.assertEqual(raised["state"], "举手")
        self.assertEqual(raised["head"]["status"], "skipped_hand_raise")
        sample = points(False)
        self.assertEqual(pose_behaviors.classify(sample, 200, 200, .8, .9)['state'], '抬头')
        self.assertEqual(pose_behaviors.classify(sample, 200, 200, .8, 1)['state'], '无法判断')
        self.assertEqual(pose_behaviors.classify(points(), 200, 200, .8, 1)['state'], '举手')
        for down, up in ((.2, .2), (.3, .1), (float('nan'), .2), (-1.01, .2)):
            with self.assertRaises(ValueError):
                pose_behaviors.classify(sample, 200, 200, down, up)


class BehaviorDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.dataset_root = root / "datasets"
        for module in (annotations, dataset_store, vision):
            p = patch.object(module, "DATASETS_DIR", self.dataset_root)
            p.start(); self.addCleanup(p.stop)
        p = patch.object(history, "HISTORY_DIR", root / "history")
        p.start(); self.addCleanup(p.stop)
        source = root / "东教室测试.png"
        Image.new("RGB", (200, 200), "white").save(source)
        rid, snapshot = history.begin_record(source)
        detections = [{"class_id": 0, "class_name": "person", "confidence": confidence,
                       "bbox": [50, 30, 150, 190], "keypoints": points(raised)}
                      for confidence, raised in ((0.9, True), (0.8, False), (0.7, True))]
        self.record = history.complete_record(rid, snapshot, {"mode": "pose", "weight": "yolov8m-pose.pt", "detections": detections,
                                                               "count": 3, "pose": pose_support.metadata()}, {"confidence": 0.7})
        self.client = TestClient(app.app)

    def test_import_annotate_export_and_repeat_preserves_manual_labels(self):
        imported = self.client.post('/api/behavior-datasets/import', json={"dataset": "actions"}).json()
        self.assertEqual(imported["progress"]["total"], 2)
        self.assertEqual(imported["progress"]["counts"]["举手"], 1)
        image = next((self.dataset_root / "actions" / "images" / "train").iterdir())
        rel = f'actions/train/{image.name}'
        opened = self.client.get('/api/annotate/image/'+rel).json()
        self.assertTrue(opened["behavior"])
        self.assertEqual([d["class_name"] for d in opened["detections"]], ["举手", "待标注"])
        payload = [{"class_id": d["class_id"], "bbox": d["bbox"], "sample_id": d["sample_id"]} for d in opened["detections"]]
        payload[1]["class_id"] = 1
        saved = self.client.post('/api/annotate/save/'+rel, json={"detections": payload})
        self.assertEqual(saved.status_code, 200, saved.text)
        exported = self.client.get('/api/behavior-datasets/actions/export').json()
        labeled = exported["images"][0]["detections"][1]
        self.assertEqual(labeled["label"], "低头")
        self.assertEqual(labeled["label_source"], "human")
        self.assertEqual(labeled["keypoints"], opened["detections"][1]["keypoints"])
        self.assertEqual(exported["progress"]["labeled"], 2)
        self.assertEqual(exported["images"][0]["source_record_id"], self.record["record_id"])
        again = self.client.post('/api/behavior-datasets/import', json={"dataset": "actions"}).json()
        self.assertEqual(again["skipped"], 1)
        reopened = annotations.read(image)
        assert reopened is not None
        self.assertEqual(reopened["detections"][1]["label"], "低头")
        self.assertEqual(self.client.put('/api/datasets/actions/classes', json={"classes": ["bad"]}).status_code, 400)
        self.assertEqual(self.client.post('/api/annotate/prelabel/'+rel).status_code, 400)

    def test_invalid_saves_do_not_change_labels_or_keypoints(self):
        annotations.import_records("actions", [self.record["record_id"]])
        image = next((self.dataset_root / "actions" / "images" / "train").iterdir())
        before = copy.deepcopy(annotations.read(image))
        assert before is not None
        dets = copy.deepcopy(before["detections"])
        dets[0]["sample_id"] = "unknown"
        with self.assertRaises(ValueError):
            annotations.save(image, dets)
        self.assertEqual(annotations.read(image), before)
        with self.assertRaises(ValueError):
            annotations.import_records("../outside", [self.record["record_id"]])


if __name__ == "__main__":
    unittest.main()
