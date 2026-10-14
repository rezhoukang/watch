import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import app
import head_fusion
import inference_history
import rtdetrv4
from web import common


def detection(box, name="head", confidence=0.25, class_id=0):
    return {"bbox": box, "class_name": name, "class_id": class_id, "confidence": confidence}


class HeadFusionTests(unittest.TestCase):
    def test_confidence_gate_in_both_directions(self):
        baseline = {"detections": [detection([0, 0, 10, 10], confidence=0.1)]}
        candidates = {"detections": [
            detection([5, 0, 15, 10], confidence=0.49),
            detection([5, 0, 15, 10], confidence=0.5),
            detection([5, 0, 15, 10], confidence=0.51),
            detection([20, 0, 30, 10], confidence=0.1),
        ]}
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                result = head_fusion.fuse_results(candidates if reverse else baseline, baseline if reverse else candidates,
                                                  reverse, merge_iou=0.34, merge_confidence=0.5)
                self.assertEqual([d["confidence"] for d in result["detections"]], [0.1, 0.51, 0.1])
                self.assertEqual(result["fusion"]["discarded_overlaps"], 2)
                self.assertEqual(result["fusion"]["merge_confidence"], 0.5)

    def test_adjustable_iou_boundary_in_both_directions(self):
        yolo = {"detections": [detection([0, 0, 10, 10], "flag", 0.8)]}
        heads = {"detections": [detection([5, 0, 15, 10], class_id=1)]}
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                strict = head_fusion.fuse_results(yolo, heads, reverse, merge_iou=0.3, merge_confidence=0)
                boundary = head_fusion.fuse_results(yolo, heads, reverse, merge_iou=1/3, merge_confidence=0)
                self.assertEqual(strict["count"], 1)
                self.assertEqual(boundary["count"], 2)
                self.assertEqual(boundary["fusion"]["merge_iou"], 1/3)
                self.assertEqual(strict["detections"][0]["source_model"], rtdetrv4.WEIGHTS_NAME if reverse else "medium.pt")

    def test_rtdetr_baseline_keeps_overlap_and_adds_only_disjoint_yolo(self):
        yolo = {"detections": [detection([0, 0, 10, 10], "flag", 0.8), detection([20, 0, 30, 10], "head", 0.9)]}
        heads = {"detections": [detection([1, 1, 9, 9], class_id=1)]}
        result = head_fusion.fuse_results(yolo, heads, rtdetr_base=True)
        self.assertEqual([det["bbox"] for det in result["detections"]], [[1, 1, 9, 9], [20, 0, 30, 10]])
        self.assertEqual([det["source_model"] for det in result["detections"]], [rtdetrv4.WEIGHTS_NAME, "medium.pt"])
        self.assertEqual(result["fusion"]["discarded_overlaps"], 1)
        self.assertEqual(result["fusion"]["base_count"], 1)
        self.assertEqual(result["fusion"]["incoming_count"], 2)
        self.assertEqual(result["weight"], rtdetrv4.WEIGHTS_NAME)
        self.assertEqual(head_fusion.parameters(True)["base"]["confidence"], 0.25)
        self.assertEqual(head_fusion.parameters(True)["incoming"]["confidence"], 0.45)

    def test_api_saves_reversed_direction_and_colors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (40, 30), "white").save(root / "test.png")
            with patch.object(common, "UPLOAD_DIR", root), patch.object(common, "_require_image_backend"), \
                 patch.object(inference_history, "HISTORY_DIR", root / "history"), \
                 patch.object(head_fusion.vision, "detect_image", side_effect=[
                     {"detections": [detection([0, 0, 10, 10], confidence=0.8)]},
                     {"detections": [detection([20, 0, 30, 10], class_id=1)]},
                 ]):
                response = TestClient(app.app).post("/api/detect/image/test.png?head_detection=true&rtdetr_base=true&merge_iou=0.3&merge_confidence=0.5")
                self.assertEqual(response.status_code, 200, response.text)
                record = response.json()
                self.assertEqual(record["parameters"], head_fusion.parameters(True, 0.3, 0.5))
                self.assertEqual(record["fusion"]["merge_confidence"], 0.5)
                self.assertEqual(record["fusion"]["merge_iou"], 0.3)
                self.assertTrue(record["fusion"]["rtdetr_base"])
                with Image.open(inference_history.record_dir(record["record_id"]) / "result.png") as image:
                    self.assertEqual(image.convert("RGB").getpixel((20, 0)), (0, 187, 102))
                    self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 153, 0))

    def test_zero_iou_additions_preserve_baseline_and_classes(self):
        base = {"detections": [detection([0, 0, 10, 10], "flag", 0.8)], "classes_used": ["flag"]}
        incoming = {"detections": [
            detection([9.999, 0, 20, 10]),  # Any positive overlap, even across classes, loses.
            detection([10, 0, 20, 10]),     # Edge touching: IoU = 0, retain.
            detection([10, 0, 20, 10]),     # Incoming boxes do not suppress each other.
        ]}
        original = copy.deepcopy((base, incoming))
        with patch.object(head_fusion.vision, "detect_image", side_effect=[base, incoming]) as detect:
            result = head_fusion.detect_image(Path("image.png"))
        self.assertEqual((base, incoming), original)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["detections"][0]["bbox"], [0, 0, 10, 10])
        self.assertEqual(result["summary"], {"flag": 1, "head": 2})
        self.assertEqual(result["fusion"]["discarded_overlaps"], 1)
        self.assertEqual(result["detections"][1]["class_id"], 1)
        self.assertEqual(result["detections"][1]["source_class_id"], 0)
        self.assertEqual([(c.kwargs["weights"], c.kwargs["conf"], c.kwargs["imgsz"]) for c in detect.call_args_list],
                         [("medium.pt", 0.45, 640), (rtdetrv4.WEIGHTS_NAME, 0.25, 640)])

    def test_empty_baseline_retains_all_heads(self):
        with patch.object(head_fusion.vision, "detect_image", side_effect=[
            {"detections": [], "classes_used": ["head"]},
            {"detections": [detection([0, 0, 10, 10])]},
        ]):
            result = head_fusion.detect_image(Path("image.png"))
        self.assertEqual(result["person_count"], 1)
        self.assertEqual(result["fusion"]["added_count"], 1)

    def test_api_fixed_parameters_and_saved_fusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (40, 30), "white").save(root / "课堂.png")
            with patch.object(common, "UPLOAD_DIR", root), \
                 patch.object(common, "_require_image_backend") as ready, \
                 patch.object(inference_history, "HISTORY_DIR", root / "history"), \
                 patch.object(head_fusion.vision, "detect_image", side_effect=[
                     {"detections": [detection([0, 0, 10, 10], confidence=0.8)], "classes_used": ["head"]},
                     {"detections": [detection([20, 0, 30, 10], class_id=1)]},
                 ]) as detect:
                client = TestClient(app.app)
                response = client.post("/api/detect/image/课堂.png", params={
                    "head_detection": True, "imgsz": 1280, "weights": "unused.pt", "conf": 0.9,
                })
                self.assertEqual(response.status_code, 200, response.text)
                record = response.json()
                self.assertEqual(record["kind"], "merge")
                self.assertEqual(record["imgsz"], 640)
                self.assertEqual(record["count"], 2)
                self.assertEqual(record["parameters"], head_fusion.parameters())
                self.assertEqual([c.args[0] for c in ready.call_args_list], ["medium.pt", rtdetrv4.WEIGHTS_NAME])
                self.assertEqual([c.kwargs["conf"] for c in detect.call_args_list], [0.45, 0.25])
                saved = client.get(record["record_url"]).json()
                self.assertEqual(saved["detections"], record["detections"])
                self.assertIn("fusion", client.get("/api/inferences").json()["records"][0])
                with Image.open(inference_history.record_dir(record["record_id"]) / "result.png") as image:
                    pixels = image.convert("RGB")
                    self.assertEqual(pixels.getpixel((0, 0)), (0, 187, 102))
                    self.assertEqual(pixels.getpixel((20, 0)), (255, 153, 0))


if __name__ == "__main__":
    unittest.main()
