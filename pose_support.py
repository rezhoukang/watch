"""COCO 17-keypoint serialization and shared skeleton rendering."""
import math
from typing import Any

KEYPOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                  "left_wrist", "right_wrist", "left_hip", "right_hip",
                  "left_knee", "right_knee", "left_ankle", "right_ankle"]
SKELETON = [[5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
DISPLAY_KEYPOINTS = list(range(11))
KEYPOINT_CONFIDENCE = 0.5


def metadata() -> dict:
    return {"keypoint_names": KEYPOINT_NAMES, "skeleton": SKELETON,
            "display_keypoints": DISPLAY_KEYPOINTS, "display_body": "upper",
            "keypoint_confidence": KEYPOINT_CONFIDENCE, "coordinates": "original_image_xy_confidence"}


def serialize_keypoints(result: Any) -> list[list[list[float]]]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return []
    poses = []
    for person in keypoints.data.cpu().tolist():
        if len(person) != 17:
            raise ValueError("Pose 骨架需要 COCO 格式的 17 个关键点")
        points = []
        for point in person:
            x, y = float(point[0]), float(point[1])
            confidence = float(point[2]) if len(point) > 2 else 1.0
            points.append([round(x, 2), round(y, 2), round(confidence, 4)]
                          if all(math.isfinite(v) for v in (x, y, confidence)) else [0.0, 0.0, 0.0])
        poses.append(points)
    return poses


def visible(point: list, width: int, height: int) -> bool:
    return len(point) >= 3 and all(math.isfinite(v) for v in point[:3]) and (
        point[2] >= KEYPOINT_CONFIDENCE and 0 < point[0] < width and 0 < point[1] < height)


def draw_pil(image, detections: list[dict]) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for detection in detections:
        points = detection.get("keypoints", [])
        if len(points) != 17:
            continue
        for a, b in SKELETON:
            if visible(points[a], image.width, image.height) and visible(points[b], image.width, image.height):
                draw.line([tuple(points[a][:2]), tuple(points[b][:2])], fill="#00bcd4", width=2)
        for point in points[:11]:
            if visible(point, image.width, image.height):
                x, y = point[:2]
                draw.ellipse((x-2, y-2, x+2, y+2), fill="#ff4081")


def draw_cv2(frame, poses: list[list[list[float]]]) -> None:
    import cv2

    height, width = frame.shape[:2]
    for points in poses:
        for a, b in SKELETON:
            if visible(points[a], width, height) and visible(points[b], width, height):
                cv2.line(frame, (int(points[a][0]), int(points[a][1])),
                         (int(points[b][0]), int(points[b][1])), (212, 188, 0), 2)
        for point in points[:11]:
            if visible(point, width, height):
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (129, 64, 255), -1)
