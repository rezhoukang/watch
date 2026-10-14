import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import cv2
from PIL import Image
from fastapi.testclient import TestClient

import app
import inference_history
import pose_support
import pose_roi
import vision
from ultralytics.engine.results import Results
from web import common


def prediction():
    points: list[list[float]] = [[0, 0, 0] for _ in range(17)]
    points[5], points[7], points[9] = [30, 40, 0.9], [50, 60, 0.8], [70, 80, 0.2]
    points[0], points[6] = [55, 29, .9], [80, 40, .9]
    points[11], points[13] = [120, 70, 0.9], [140, 85, 0.9]
    return Results(np.full((100, 200, 3), 255, dtype=np.uint8), path='pose.png', names={0: 'person'},
                   boxes=torch.tensor([[10, 10, 90, 90, .9, 0]], dtype=torch.float32),
                   keypoints=torch.tensor([points], dtype=torch.float32))


class PoseTests(unittest.TestCase):
    def test_roi_phone_state_and_evidence_survive_image_archive_and_video_render(self):
        person = prediction()
        assert person.keypoints is not None
        for index, point in ((7, [20, 60, .9]), (8, [90, 60, .9]), (9, [40, 80, .9]), (10, [70, 80, .9])):
            person.keypoints.data[0, index] = torch.tensor(point)
        object_result = Results(np.full((100, 200, 3), 255, dtype=np.uint8), path='full.png', names={0: 'cell phone'},
                                boxes=torch.tensor([[45, 65, 55, 75, .9, 0]]))
        roi_model = Mock()
        roi_model.predict.return_value = [object_result]
        model = Mock()
        model.predict.return_value = [person]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (200, 100), 'white').save(root / 'pose.png')
            with patch.object(common, 'UPLOAD_DIR', root), patch.object(common, '_require_image_backend'), \
                 patch.object(vision, 'get_model', return_value=model), \
                 patch.object(inference_history, 'HISTORY_DIR', root / 'history'), \
                 patch.object(pose_roi, '_get_model', return_value=roi_model):
                client = TestClient(app.app)
                response = client.post('/api/detect/image/pose.png?weights=yolov8m-pose.pt&head_down_r_threshold=.3&head_up_r_threshold=.4')
                self.assertEqual(response.status_code, 200, response.text)
                record = response.json()
                self.assertEqual(record['detections'][0]['behavior']['state'], '疑似看手机')
                self.assertEqual(record['detections'][0]['behavior']['roi']['objects'][0]['bbox'], [45, 65, 55, 75])
                saved = client.get(record['record_url']).json()
                self.assertEqual(saved['detections'], record['detections'])
                with patch.object(pose_support, 'draw_cv2') as draw:
                    vision._annotate_frame(person.orig_img.copy(), person, head_down_r_threshold=.3, head_up_r_threshold=.4)
                    self.assertEqual(draw.call_args.args[4][0]['state'], '疑似看手机')

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
                response = client.post("/api/detect/image/pose.png?weights=yolov8m-pose.pt&head_down_r_threshold=0.3&head_up_r_threshold=0.4")
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["mode"], "pose")
                self.assertEqual(result["pose"]["keypoint_names"], pose_support.KEYPOINT_NAMES)
                self.assertEqual(result["detections"][0]["keypoints"][5], [30, 40, 0.9])
                self.assertEqual(len(result["detections"][0]["keypoints"]), 17)
                self.assertEqual(result["detections"][0]["keypoints"][11], [120, 70, 0.9])
                self.assertEqual(result["pose"]["display_keypoints"], list(range(11)))
                self.assertEqual(result['detections'][0]['behavior']['state'], '低头')
                self.assertEqual(result['pose']['head_rule']['head_down_r_threshold'], .3)
                self.assertEqual(result['pose']['head_rule']['head_up_r_threshold'], .4)
                self.assertEqual(result['parameters']['head_down_r_threshold'], .3)
                saved = client.get(result["record_url"]).json()
                self.assertEqual(saved["detections"], result["detections"])
                self.assertEqual(saved['pose']['head_rule']['head_up_r_threshold'], .4)
                with Image.open(inference_history.record_dir(result["record_id"]) / "result.png") as image:
                    pixels = image.convert("RGB")
                    self.assertEqual(pixels.getpixel((40, 50)), (0, 188, 212))
                    self.assertEqual(pixels.getpixel((60, 70)), (255, 255, 255))
                    self.assertEqual(pixels.getpixel((120, 70)), (255, 255, 255))

    def test_video_annotation_draws_visible_limbs_and_skips_low_confidence(self):
        frame = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = vision._annotate_frame(frame, prediction())
        self.assertEqual(result[50, 40].tolist(), [212, 188, 0])
        self.assertEqual(result[70, 60].tolist(), [255, 255, 255])
        self.assertEqual(result[70, 120].tolist(), [255, 255, 255])
        self.assertFalse(pose_support.visible([0, 0, 0.9], 200, 100))
        self.assertFalse(pose_support.visible([300, 10, 0.9], 200, 100))

    def test_unknown_filter_removes_matching_box_and_skeleton_including_empty(self):
        good = prediction()
        assert good.keypoints is not None and good.boxes is not None
        assert isinstance(good.keypoints.data, torch.Tensor) and isinstance(good.boxes.data, torch.Tensor)
        bad_points = good.keypoints.data.clone()
        bad_points[0, 0, 2] = .59
        mixed = Results(good.orig_img, path='pose.png', names=good.names,
                        boxes=torch.cat([good.boxes.data, good.boxes.data]),
                        keypoints=torch.cat([good.keypoints.data, bad_points]))
        filtered, excluded = pose_support.filter_result(mixed)
        self.assertEqual(excluded, 1)
        self.assertEqual(len(filtered.boxes), 1)
        self.assertEqual(len(filtered.keypoints), 1)
        ambiguous, excluded = pose_support.filter_result(good, .1, .3)
        self.assertEqual(excluded, 1)
        self.assertEqual(len(ambiguous.boxes), 0)
        unknown = mixed[[1]]
        empty, excluded = pose_support.filter_result(unknown)
        self.assertEqual(excluded, 1)
        self.assertEqual(len(empty.boxes), 0)
        frame = np.full((100, 200, 3), 255, dtype=np.uint8)
        self.assertTrue(np.all(vision._annotate_frame(frame, unknown) == 255))

    def test_invalid_r_threshold_is_rejected_by_all_inference_routes(self):
        client = TestClient(app.app)
        for path in ('/api/detect/image/pose.png', '/api/detect/video/pose.mp4', '/api/annotate/prelabel/test/train/pose.png'):
            for value in ('1.01', '-1.01', 'nan'):
                self.assertEqual(client.post(path+'?head_down_r_threshold='+value).status_code, 422)
                self.assertEqual(client.post(path+'?head_up_r_threshold='+value).status_code, 422)
            for query in ('head_down_r_threshold=.2&head_up_r_threshold=.2', 'head_down_r_threshold=.3&head_up_r_threshold=.1'):
                self.assertEqual(client.post(path+'?'+query).status_code, 422)

    def test_video_ambiguous_people_are_excluded_from_curve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'pose.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 5, (200, 100))  # type: ignore[attr-defined]
            self.assertTrue(writer.isOpened())
            for _ in range(2):
                writer.write(np.full((100, 200, 3), 255, dtype=np.uint8))
            writer.release()
            model = Mock()
            model.task = 'pose'
            model.predict.return_value = [prediction()]
            job_id = 'pose-ambiguous-test'
            vision.JOBS[job_id] = {'status': 'running', 'head_counts': [],
                                   'parameters': {'head_down_r_threshold': .1, 'head_up_r_threshold': .3}}
            self.addCleanup(vision.JOBS.pop, job_id, None)
            with patch.object(vision, 'OUTPUT_DIR', root), patch.object(vision, 'get_model', return_value=model), \
                 patch.object(vision, '_save_best_frame', return_value='best.jpg'):
                vision._run_video_job(job_id, source, 'yolov8m-pose.pt', .7, 1,
                                      head_down_r_threshold=.1, head_up_r_threshold=.3)
            job = vision.JOBS[job_id]
            self.assertEqual(job['status'], 'done', job.get('message'))
            self.assertEqual(job['head_counts'], [[0, 0], [1, 0]])
            self.assertIn('_pose_down0.1_up0.3', job['result_url'])


if __name__ == "__main__":
    unittest.main()
