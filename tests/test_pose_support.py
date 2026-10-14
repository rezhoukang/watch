import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

import app
import inference_history
import pose_support
import vision
from web import common


def prediction():
    points: list[list[float]] = [[0, 0, 0] for _ in range(17)]
    points[5], points[7], points[9] = [30, 40, 0.9], [50, 60, 0.8], [70, 80, 0.2]
    data = Mock()
    data.cpu.return_value.tolist.return_value = [points]
    return SimpleNamespace(names={0: "person"}, orig_shape=(100, 200),
                           keypoints=SimpleNamespace(data=data),
                           boxes=[SimpleNamespace(cls=[0], conf=[0.9], xyxy=[np.array([10, 10, 90, 90])])])


class PoseTests(unittest.TestCase):
    def test_image_api_preserves_keypoints_and_archives_visible_skeleton(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (200, 100), "white").save(root / "pose.png")
            model = Mock()
            model.predict.return_value = [prediction()]
            with patch.object(common, "UPLOAD_DIR", root), patch.object(common, "_require_image_backend"), \
                 patch.object(vision, "get_model", return_value=model), \
                 patch.object(inference_history, "HISTORY_DIR", root / "history"):
                client = TestClient(app.app)
                response = client.post("/api/detect/image/pose.png?weights=yolov8m-pose.pt")
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["mode"], "pose")
                self.assertEqual(result["pose"]["keypoint_names"], pose_support.KEYPOINT_NAMES)
                self.assertEqual(result["detections"][0]["keypoints"][5], [30, 40, 0.9])
                self.assertEqual(len(result["detections"][0]["keypoints"]), 17)
                saved = client.get(result["record_url"]).json()
                self.assertEqual(saved["detections"], result["detections"])
                with Image.open(inference_history.record_dir(result["record_id"]) / "result.png") as image:
                    pixels = image.convert("RGB")
                    self.assertEqual(pixels.getpixel((40, 50)), (0, 188, 212))
                    self.assertEqual(pixels.getpixel((60, 70)), (255, 255, 255))

    def test_video_annotation_draws_visible_limbs_and_skips_low_confidence(self):
        frame = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = vision._annotate_frame(frame, prediction())
        self.assertEqual(result[50, 40].tolist(), [212, 188, 0])
        self.assertEqual(result[70, 60].tolist(), [255, 255, 255])
        self.assertFalse(pose_support.visible([0, 0, 0.9], 200, 100))
        self.assertFalse(pose_support.visible([300, 10, 0.9], 200, 100))


if __name__ == "__main__":
    unittest.main()
