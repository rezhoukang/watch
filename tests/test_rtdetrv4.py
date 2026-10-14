import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import app
from web import common
import rtdetrv4
import vision
import inference_history


class HeadDetectionTests(unittest.TestCase):
    def test_count_only_heads_and_clip_to_original_image(self):
        detections = rtdetrv4._head_detections(
            [0, 1, 1, 1, 1, 1],
            [[0, 0, 40, 80], [-5, 10, 110, 70], [1, 2, 10, 12],
             [8, 4, 2, 10], [float("nan"), 0, 10, 10], [0, 0, 10, 10]],
            [0.99, 0.8, 0.2, 0.9, 0.9, float("nan")], 100, 60, 0.38,
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_id"], 1)
        self.assertEqual(detections[0]["bbox"], [0, 10, 100, 60])
        self.assertEqual(detections[0]["bbox_cxcywh"], [50, 35, 100, 50])

    def test_unicode_image_output_and_zero_count(self):
        class FakeDetector:
            device = "cpu"

            def predict(self, image, size):
                self.image_size = image.size
                self.input_size = size
                return [0, 1], [[1, 2, 30, 40], [5, 5, 15, 20]], [0.9, 0.5]

        fake = FakeDetector()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            src = directory / "教室.png"
            Image.new("RGB", (120, 80)).save(src)
            output = directory / "results"
            with patch.object(rtdetrv4, "_detector", fake), patch.object(vision, "RESULTS_DIR", output):
                result = vision.detect_image(src, weights=rtdetrv4.WEIGHTS_NAME, imgsz=641)
                self.assertEqual(result["person_count"], 1)
                self.assertEqual(result["summary"], {"head": 1})
                self.assertEqual((fake.image_size, fake.input_size), ((120, 80), 640))
                self.assertTrue((output / result["result"]).is_file())
                empty_output = directory / "unsaved"
                result = rtdetrv4.detect_image(src, 0.8, 640, empty_output, save_result=False)
                self.assertEqual(result["person_count"], 0)
                self.assertEqual(result["detections"], [])
                self.assertEqual(result["result_url"], "")
                self.assertFalse(empty_output.exists())

    def test_preprocessing_uses_rgb_square_and_width_height(self):
        import torch

        detector = rtdetrv4._HeadDetector.__new__(rtdetrv4._HeadDetector)
        detector.device = "cpu"
        captured = {}

        def model(tensor):
            captured["tensor"] = tensor
            return {}

        def postprocessor(outputs, size):
            captured["size"] = size.tolist()
            return torch.tensor([[1]]), torch.tensor([[[10, 20, 30, 40]]]), torch.tensor([[0.9]])

        detector.model = model
        detector.postprocessor = postprocessor
        detector.predict(Image.new("RGB", (960, 540), (255, 0, 0)), 320)
        self.assertEqual(tuple(captured["tensor"].shape), (1, 3, 320, 320))
        self.assertEqual(captured["size"], [[960, 540]])
        self.assertEqual(captured["tensor"][0, :, 0, 0].tolist(), [1.0, 0.0, 0.0])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app.app)

    def test_head_backend_is_independent_of_yolo(self):
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp) / "classroom.png"
            Image.new("RGB", (160, 90)).save(src)
            ready = {"available": True, "message": "ready"}
            with patch.object(common, "UPLOAD_DIR", src.parent), patch.object(rtdetrv4, "status", return_value=ready), \
                 patch.object(inference_history, "HISTORY_DIR", src.parent / "history"), \
                 patch.object(vision, "yolo_available", return_value=(False, "missing")), \
                 patch.object(vision, "detect_image", return_value={"mode": "rtdetrv4", "person_count": 3,
                     "detections": [{"bbox": [1, 1, 10, 10]}] * 3}) as detect:
                response = self.client.post(f"/api/detect/image/{src.name}", params={"weights": rtdetrv4.WEIGHTS_NAME})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["person_count"], 3)
                self.assertEqual(detect.call_args.kwargs["weights"], rtdetrv4.WEIGHTS_NAME)

    def test_missing_backend_returns_actionable_503(self):
        with patch.object(rtdetrv4, "status", return_value={"available": False, "message": "install dependencies"}):
            response = self.client.post("/api/detect/image/test.png", params={"weights": rtdetrv4.WEIGHTS_NAME})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "install dependencies")

    def test_model_supports_image_and_video_but_not_yolo_training(self):
        with patch.object(vision, "yolo_available", return_value=(True, "ready")), \
             patch.object(vision, "clip_available", return_value=True):
            state = self.client.get("/api/yolo/status").json()
        self.assertIn(rtdetrv4.WEIGHTS_NAME, state["image_weights"])
        self.assertIn(rtdetrv4.WEIGHTS_NAME, state["video_weights"])
        self.assertNotIn(rtdetrv4.WEIGHTS_NAME, state["weights"])
        for route in ["/api/train",
                      "/api/annotate/prelabel/example/train/test.png"]:
            with self.subTest(route=route):
                response = self.client.post(route, params={"weights": rtdetrv4.WEIGHTS_NAME, "dataset": "example"})
                self.assertEqual(response.status_code, 400)

    def test_yolo_image_still_uses_yolo_readiness(self):
        with patch.object(vision, "yolo_available", return_value=(False, "YOLO missing")):
            response = self.client.post("/api/detect/image/test.png", params={"weights": "medium.pt"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("YOLO missing", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
