import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app
import head_fusion
import vision
from web import common


class VideoHeadFusionTests(unittest.TestCase):
    def test_video_endpoint_passes_fusion_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "test.mp4"
            src.touch()
            with patch.object(common, "VIDEO_DIR", src.parent), patch.object(common, "_require_head_detection"), \
                 patch.object(vision, "yolo_available", return_value=(True, "ready")), \
                 patch.object(vision, "world_ready", return_value=(True, "ready")), \
                 patch.object(vision, "start_video_job", return_value="example") as start:
                response = TestClient(app.app).post("/api/detect/video/test.mp4", params={
                    "head_detection": True, "weights": "ignored.pt", "conf": 0.99, "stride": 2,
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(response.json()["head_detection"])
                self.assertTrue(start.call_args.kwargs["head_detection"])
                self.assertEqual(start.call_args.kwargs["weights"], "medium.pt")
                self.assertEqual(start.call_args.kwargs["conf"], 0.45)
                self.assertEqual(start.call_args.kwargs["stride"], 2)

    def test_video_worker_exports_fused_coordinates_counts_and_preview_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "sample.avi"
            writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 5, (96, 64))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.full((64, 96, 3), 255, dtype=np.uint8))
            writer.release()
            fused = head_fusion.fuse_results({"classes_used": ["head"], "detections": [
                {"bbox": [5, 5, 15, 15], "class_id": 0, "class_name": "head", "confidence": 0.8},
            ]}, {"detections": [
                {"bbox": [25, 5, 35, 15], "class_id": 1, "class_name": "head", "confidence": 0.2},
            ]})
            job_id = "video-test"
            vision.JOBS[job_id] = {"status": "running", "head_counts": [], "head_detection": True,
                                   "parameters": head_fusion.parameters()}
            self.addCleanup(vision.JOBS.pop, job_id, None)
            output = root / "outputs"
            output.mkdir()
            with patch.object(vision, "OUTPUT_DIR", output), patch.object(common, "OUTPUT_DIR", output), \
                 patch.object(vision, "VIDEO_DIR", root), patch.object(common, "VIDEO_DIR", root), \
                 patch.object(common, "UPLOAD_DIR", root), patch.object(vision, "device_arg", return_value="cpu"), \
                 patch.object(vision, "get_model"), patch.object(vision, "_save_best_frame", return_value="best.jpg"), \
                 patch.object(head_fusion, "detect_frame", return_value=fused) as detect:
                vision._run_video_job(job_id, src, "medium.pt", 0.45, 2, head_detection=True)
                job = vision.JOBS[job_id]
                self.assertEqual(job["status"], "done", job.get("message"))
                self.assertEqual(detect.call_count, 3)
                self.assertEqual(job["head_counts"], [[0, 2], [2, 2], [4, 2]])
                client = TestClient(app.app)
                response = client.get(job["detections_url"])
                self.assertEqual(response.status_code, 200)
                lines = [json.loads(line) for line in response.text.splitlines()]
                self.assertEqual(lines[0]["parameters"], head_fusion.parameters())
                self.assertEqual([line["frame_index"] for line in lines[1:]], [0, 2, 4])
                self.assertEqual(lines[1]["detections"][1]["bbox"], [25, 5, 35, 15])
                self.assertEqual(lines[1]["fusion"]["added_count"], 1)
                metadata = client.get(job["chart_url"]).json()
                self.assertTrue(metadata["head_detection"])
                listed = client.get("/api/videos").json()["outputs"][0]
                self.assertTrue(listed["head_detection"])
                self.assertEqual(listed["detections_url"], job["detections_url"])
                self.assertEqual(client.delete("/api/outputs/" + Path(job["result_url"]).name).status_code, 200)
                self.assertFalse((output / "detections" / Path(job["detections_url"]).name).exists())

    def test_frame_uses_same_pixels_fixed_parameters_and_merge(self):
        box = SimpleNamespace(cls=[0], conf=[0.9], xyxy=[np.array([0, 0, 10, 10])])
        model = Mock()
        model.predict.return_value = [Mock(names={0: "head"}, boxes=[box])]
        bgr = np.zeros((20, 30, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255
        with patch.object(vision, "device_arg", return_value="cpu"), patch.object(head_fusion.rtdetrv4, "detect_pil", return_value=([
            {"class_id": 1, "class_name": "head", "confidence": 0.2, "bbox": [5, 5, 15, 15]},
            {"class_id": 1, "class_name": "head", "confidence": 0.2, "bbox": [10, 0, 20, 10]},
        ], "cpu")) as heads:
            result = head_fusion.detect_frame(bgr, model)
        self.assertEqual(model.predict.call_args.kwargs["imgsz"], 640)
        self.assertEqual(model.predict.call_args.kwargs["conf"], 0.45)
        self.assertIs(model.predict.call_args.kwargs["source"], bgr)
        self.assertEqual(heads.call_args.args[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(heads.call_args.args[1:], (0.2, 640))
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["fusion"]["discarded_overlaps"], 1)

    def test_prelabel_fuses_then_maps_dataset_class_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            ds = Path(directory)
            src = ds / "images" / "train" / "image.png"
            src.parent.mkdir(parents=True)
            Image.new("RGB", (40, 30)).save(src)
            vision.write_dataset_classes(ds, ["flag", "head"])
            fused = head_fusion.fuse_results({"classes_used": ["head"], "detections": []}, {"detections": [
                {"bbox": [0, 0, 10, 10], "class_id": 1, "class_name": "head", "confidence": 0.2},
            ]})
            with patch.object(common, "_require_head_detection"), \
                 patch.object(vision, "resolve_annotate_image", return_value=src), \
                 patch.object(vision, "yolo_available", return_value=(True, "ready")), \
                 patch.object(vision, "world_ready", return_value=(True, "ready")), \
                 patch.object(head_fusion, "detect_image", return_value=fused):
                response = TestClient(app.app).post("/api/annotate/prelabel/test/train/image.png?head_detection=true")
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertTrue(result["head_detection"])
            self.assertEqual(result["detections"][0]["class_id"], 1)
            self.assertEqual(result["parameters"]["imgsz"], 640)


if __name__ == "__main__":
    unittest.main()
