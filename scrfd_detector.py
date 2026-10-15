"""SCRFD face boxes and five landmarks only; no identity features or matching."""
from pathlib import Path

import cv2
import numpy as np

from project_paths import MODELS_DIR

MODEL_PATH = MODELS_DIR / "scrfd" / "det_10g.onnx"
MODEL_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
MODEL_SOURCE = "https://huggingface.co/immich-app/buffalo_l/resolve/main/detection/model.onnx"


class ScrfdDetector:
    def __init__(self, path: Path = MODEL_PATH):
        import onnxruntime as ort

        if not path.is_file():
            raise FileNotFoundError(f"SCRFD 检测权重不存在：{path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        if len(outputs) != 9:
            raise ValueError("需要三层、含五个人脸关键点的 SCRFD 检测模型（9个输出）")
        self.output_names = [v.name for v in outputs]
        self.centers = {}

    def detect(self, image, threshold: float = 0.5) -> list[dict]:
        height, width = image.shape[:2]
        if min(height, width) < 2:
            return []
        scale = min(640/width, 640/height)
        resized_w, resized_h = max(1, int(width*scale)), max(1, int(height*scale))
        canvas = np.zeros((640, 640, 3), dtype=np.uint8)
        canvas[:resized_h, :resized_w] = cv2.resize(image, (resized_w, resized_h))
        blob = cv2.dnn.blobFromImage(canvas, 1/128, (640, 640), (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        faces = []
        for layer, stride in enumerate((8, 16, 32)):
            scores = outputs[layer].reshape(-1)
            boxes = outputs[layer+3].reshape(-1, 4)*stride
            landmarks = outputs[layer+6].reshape(-1, 5, 2)*stride
            if stride not in self.centers:
                grid = np.indices((640//stride, 640//stride))[::-1].reshape(2, -1).T*stride
                self.centers[stride] = np.repeat(grid, 2, axis=0).astype(np.float32)
            centers = self.centers[stride]
            if len(scores) != len(centers):
                raise ValueError("SCRFD 输出网格与640输入不匹配")
            for i in np.flatnonzero(scores >= threshold):
                x, y = centers[i]
                left, top, right, bottom = boxes[i]
                bbox = np.array([x-left, y-top, x+right, y+bottom])/scale
                points = (landmarks[i]+centers[i])/scale
                if not np.isfinite(bbox).all() or not np.isfinite(points).all():
                    continue
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height)
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                faces.append({"bbox": bbox.tolist(), "landmarks": points.tolist(), "confidence": float(scores[i])})
        if not faces:
            return []
        rectangles = [[v['bbox'][0], v['bbox'][1], v['bbox'][2]-v['bbox'][0], v['bbox'][3]-v['bbox'][1]] for v in faces]
        indices = np.asarray(cv2.dnn.NMSBoxes(rectangles, [v['confidence'] for v in faces], threshold, .4)).reshape(-1)
        return [faces[int(i)] for i in indices]


def assess_quality(image, face: dict, min_size: int = 24, min_sharpness: float = 50,
                   min_confidence: float = .8) -> tuple[list[str], dict]:
    """Conservative geometric/clarity heuristics; not a true 3-D pose estimate.

    All three gates are parameters. `min_confidence` defaults to 0.8 (the
    anonymous-face review baseline); callers whose detector already ran at a
    lower threshold should pass that threshold instead of double-gating.
    """
    x1, y1, x2, y2 = face['bbox']
    points = np.array(face['landmarks'], dtype=np.float64)
    reasons = []
    if face['confidence'] < min_confidence:
        reasons.append("人脸置信度不足")
    if min(x2-x1, y2-y1) < min_size:
        reasons.append("原始人脸尺寸过小")
    crop = image[max(0, int(y1)):min(image.shape[0], int(np.ceil(y2))), max(0, int(x1)):min(image.shape[1], int(np.ceil(x2)))]
    sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if crop.size else 0.0
    if sharpness < min_sharpness:
        reasons.append("人脸模糊")
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return reasons+["人脸关键点无效"], {"sharpness": round(sharpness, 2)}
    eyes = points[1]-points[0]
    eye_distance = float(np.linalg.norm(eyes))
    if eye_distance < max(2, .15*(x2-x1)):
        reasons.append("双眼位置不可靠或侧脸明显")
        return reasons, {"sharpness": round(sharpness, 2), "eye_distance": round(eye_distance, 2)}
    horizontal = eyes/eye_distance
    vertical = np.array([-horizontal[1], horizontal[0]])
    eye_center = (points[0]+points[1])/2
    yaw_proxy = abs(float(np.dot(points[2]-eye_center, horizontal)))/eye_distance
    roll = abs(float(np.degrees(np.arctan2(eyes[1], eyes[0]))))
    nose_depth = float(np.dot(points[2]-eye_center, vertical))/eye_distance
    mouth_depth = float(np.dot((points[3]+points[4])/2-eye_center, vertical))/eye_distance
    if yaw_proxy > .45:
        reasons.append("侧脸几何偏差过大")
    if roll > 30:
        reasons.append("人脸倾斜过大")
    if not (.05 < nose_depth < 1.3 and nose_depth < mouth_depth < 2):
        reasons.append("五点几何关系不可靠")
    if any(not (x1-2 <= x <= x2+2 and y1-2 <= y <= y2+2) for x, y in points):
        reasons.append("人脸关键点超出脸框")
    return reasons, {"sharpness": round(sharpness, 2), "eye_distance": round(eye_distance, 2),
                     "side_face_proxy": round(yaw_proxy, 4), "roll_degrees": round(roll, 2)}


def align_face(image, landmarks):
    reference = np.array([[.32, .36], [.68, .36], [.5, .55], [.36, .75], [.64, .75]], dtype=np.float32)*224
    matrix, _ = cv2.estimateAffinePartial2D(np.array(landmarks, dtype=np.float32), reference, method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("无法估计人脸对齐变换")
    return cv2.warpAffine(image, matrix, (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
