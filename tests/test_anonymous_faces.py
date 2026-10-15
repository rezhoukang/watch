import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

import anonymous_faces
import scrfd_detector


class AnonymousFaceTests(unittest.TestCase):
    def test_selects_only_classroom_names_and_rejects_duplicate_image_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('a教室 (1).png', 'a教室 (2).png', 'b教室 (1).png', 'abcd.png', '东教室.png'):
                (root / name).touch()
            self.assertEqual([v[0] for v in anonymous_faces.select_classroom_images(root)], ['A01', 'A02', 'B01'])
            (root / 'A教室 (1).jpg').touch()
            with self.assertRaises(ValueError):
                anonymous_faces.select_classroom_images(root)

    def test_roi_coordinates_quality_skip_and_anonymous_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.random.default_rng(7).integers(0, 255, (100, 200, 3), dtype=np.uint8)
            source = root / 'a教室 (1).png'
            anonymous_faces.save_image(source, image)
            detector = Mock()
            detector.detect.side_effect = [[{'bbox': [14, 16, 54, 66], 'confidence': .95,
                'landmarks': [[22, 29], [46, 29], [34, 40], [26, 52], [42, 52]]}], []]
            prediction = {'detections': [
                {'bbox': [40, 20, 100, 90], 'class_name': 'head', 'confidence': .9},
                {'bbox': [120, 30, 180, 90], 'class_name': 'head', 'confidence': .8},
                {'bbox': [0, 0, 200, 100], 'class_name': 'person', 'confidence': .99}]}
            with patch.object(anonymous_faces.head_fusion, 'detect_image', return_value=prediction):
                result = anonymous_faces.process_image(source, 'A01', root / 'result', detector)
            self.assertEqual(result['head_count'], 2)
            self.assertEqual(result['accepted_count'], 1)
            accepted, skipped = result['records']
            self.assertEqual(accepted['anonymous_id'], 'A01-001')
            self.assertEqual(accepted['face_bbox'], [45, 25, 85, 75])
            self.assertEqual(accepted['landmarks'][2], [65, 49])
            self.assertEqual(skipped['reasons'], ['未检测到可靠人脸'])
            self.assertTrue((root / 'result' / accepted['aligned_image']).is_file())
            aligned = cv2.imread(str(root / 'result' / accepted['aligned_image']))
            assert aligned is not None
            self.assertEqual(aligned.shape, (224, 224, 3))
            self.assertEqual(accepted['manual_name'], '')
            self.assertNotIn('embedding', accepted)

    def test_quality_rejects_blur_side_face_and_small_original_face(self):
        face = {'bbox': [20, 20, 80, 90], 'confidence': .95,
                'landmarks': [[35, 40], [65, 40], [50, 55], [40, 75], [60, 75]]}
        image = np.random.default_rng(8).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        self.assertEqual(scrfd_detector.assess_quality(image, face)[0], [])
        side = copy.deepcopy(face)
        side['landmarks'][2][0] = 70
        self.assertIn('侧脸几何偏差过大', scrfd_detector.assess_quality(image, side)[0])
        self.assertIn('人脸模糊', scrfd_detector.assess_quality(np.zeros_like(image), face)[0])
        small = {**face, 'bbox': [20, 20, 40, 40]}
        self.assertIn('原始人脸尺寸过小', scrfd_detector.assess_quality(image, small)[0])


if __name__ == '__main__':
    unittest.main()
