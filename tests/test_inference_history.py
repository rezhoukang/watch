import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import app
import inference_history as history
import rtdetrv4
import vision


def detection(box, confidence=0.8, name="head"):
    return {"bbox": box, "class_id": 1, "class_name": name, "confidence": confidence,
            "bbox_cxcywh": [(box[0]+box[2])/2, (box[1]+box[3])/2, box[2]-box[0], box[3]-box[1]]}


def result(detections):
    return {"mode": "rtdetrv4", "weight": rtdetrv4.WEIGHTS_NAME, "imgsz": 640,
            "count": len(detections), "person_count": len(detections), "classes_used": ["head"],
            "summary": {"head": len(detections)}, "detections": copy.deepcopy(detections)}


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.patch = patch.object(history, "HISTORY_DIR", self.directory / "history")
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.uploads = self.directory / "uploads"; self.uploads.mkdir()
        self.src = self.uploads / "教室.png"
        Image.new("RGB", (40, 30), "white").save(self.src)
        self.client = TestClient(app.app)

    def save(self, detections):
        record_id, snapshot = history.begin_record(self.src)
        return history.complete_record(record_id, snapshot, result(detections), {"confidence": 0.38, "weights": rtdetrv4.WEIGHTS_NAME, "imgsz": 640})

    def test_each_inference_has_immutable_original_image_and_metadata(self):
        with patch.object(app, "UPLOAD_DIR", self.uploads), patch.object(app, "_require_image_backend"), \
             patch.object(vision, "detect_image", return_value=result([detection([1, 1, 5, 5])])):
            first = self.client.post('/api/detect/image/' + self.src.name).json()
            Image.new("RGB", (40, 30), "black").save(self.src)
            second = self.client.post('/api/detect/image/' + self.src.name).json()
        self.assertNotEqual(first["record_id"], second["record_id"])
        self.assertNotEqual(first["input_sha256"], second["input_sha256"])
        self.src.unlink()
        original = self.client.get(first["source_url"])
        with Image.open(io.BytesIO(original.content)) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(self.client.get(first["record_url"]).json()["parameters"]["confidence"], 0.38)
        self.assertEqual(self.client.get('/api/inferences').json()["count"], 2)
        self.assertEqual(self.client.get(first["result_url"]).status_code, 200)

    def test_directed_merge_any_overlap_base_wins_and_edge_touch_survives(self):
        base = self.save([detection([0, 0, 10, 10], 0.4)])
        incoming = self.save([detection([9, 9, 12, 12], 0.99, "person"), detection([10, 0, 20, 10]), detection([22, 20, 30, 29])])
        before = history.read_record(base["record_id"])
        merged = self.client.post('/api/inferences/merge', json={"base_id": base["record_id"], "incoming_id": incoming["record_id"]}).json()
        self.assertEqual(merged["count"], 3)
        self.assertEqual(merged["detections"][0]["confidence"], 0.4)
        self.assertEqual(merged["detections"][1]["bbox"], [10, 0, 20, 10])
        self.assertEqual(merged["merge"]["discarded_overlaps"], 1)
        self.assertEqual(history.read_record(base["record_id"]), before)
        reversed_merge = history.merge_records(incoming["record_id"], base["record_id"])
        self.assertEqual(reversed_merge["detections"][0]["confidence"], 0.99)
        self.assertFalse(any(d["confidence"] == 0.4 for d in reversed_merge["detections"]))
        self.assertEqual(len(history.list_records()), 4)

    def test_merge_rejects_same_record_and_different_image_content(self):
        first = self.save([])
        Image.new("RGB", (40, 30), "black").save(self.src)
        second = self.save([])
        for incoming in (first, second):
            response = self.client.post('/api/inferences/merge', json={"base_id": first["record_id"], "incoming_id": incoming["record_id"]})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(len(history.list_records()), 2)

    def test_bad_ids_and_missing_records(self):
        self.assertEqual(self.client.get('/api/inferences/not-an-id').status_code, 400)
        self.assertEqual(self.client.get('/api/inferences/' + '0'*32).status_code, 404)


if __name__ == "__main__":
    unittest.main()
