"""Exclusive classroom behavior rules, calibrated on labeled East classroom images."""
import math

RULE = {"version": "hand_raise_v1", "keypoint_confidence": 0.6,
        "wrist_above_shoulder_ratio": 0.25, "wrist_above_elbow_ratio": 0.15,
        "wrist_above_head_ratio": 0.15}
HEAD_RULE = {"version": "head_height_v2", "keypoint_confidence": 0.6,
             "head_down_r_threshold": 0.05, "head_up_r_threshold": 0.20,
             "calibration": "east_classroom_behavior_20261014"}


def hand_raise(points: list, width: int, height: int) -> dict:
    def reliable(index):
        if index >= len(points):
            return False
        p = points[index]
        return len(p) >= 3 and all(math.isfinite(v) for v in p[:3]) and (
            p[2] >= RULE["keypoint_confidence"] and 0 < p[0] < width and 0 < p[1] < height)

    result = {"raised": False, "side": None, "status": "insufficient_keypoints", "rule": RULE.copy()}
    if not all(reliable(i) for i in (5, 6)):
        return result
    shoulder_width = math.hypot(points[5][0]-points[6][0], points[5][1]-points[6][1])
    if shoulder_width < 1:
        return result
    if reliable(0):
        head_y, head_source = points[0][1], "nose"
    elif reliable(1) and reliable(2):
        head_y, head_source = (points[1][1]+points[2][1])/2, "eyes"
    elif reliable(3) and reliable(4):
        head_y, head_source = (points[3][1]+points[4][1])/2, "ears"
    else:
        return result
    evidence, raised = {}, []
    for side, shoulder, elbow, wrist in (("left", 5, 7, 9), ("right", 6, 8, 10)):
        if not all(reliable(i) for i in (shoulder, elbow, wrist)):
            continue
        distances = {"above_shoulder": (points[shoulder][1]-points[wrist][1])/shoulder_width,
                     "above_elbow": (points[elbow][1]-points[wrist][1])/shoulder_width,
                     "above_head": (head_y-points[wrist][1])/shoulder_width}
        evidence[side] = {k: round(v, 4) for k, v in distances.items()}
        if (distances["above_shoulder"] >= RULE["wrist_above_shoulder_ratio"] and
                distances["above_elbow"] >= RULE["wrist_above_elbow_ratio"] and
                distances["above_head"] >= RULE["wrist_above_head_ratio"]):
            raised.append(side)
    result.update({"raised": bool(raised), "side": "both" if len(raised) == 2 else raised[0] if raised else None,
                   "status": "raised" if raised else "not_raised" if len(evidence) == 2 else "insufficient_keypoints",
                   "shoulder_width": round(shoulder_width, 2), "head_source": head_source, "evidence": evidence})
    return result


def validate_thresholds(head_down_r_threshold: float, head_up_r_threshold: float) -> None:
    if any(not math.isfinite(v) or not -1 <= v <= 1 for v in (head_down_r_threshold, head_up_r_threshold)):
        raise ValueError("低头和抬头 R 阈值必须在 -1 到 1 之间")
    if head_down_r_threshold >= head_up_r_threshold:
        raise ValueError("低头 R 阈值必须小于抬头 R 阈值")


def classify(points: list, width: int, height: int, head_down_r_threshold: float = 0.05,
             head_up_r_threshold: float = 0.20) -> dict:
    validate_thresholds(head_down_r_threshold, head_up_r_threshold)
    raised = hand_raise(points, width, height)
    result = {"state": "无法判断", "hand_raise": raised,
              "head": {"rule": {**HEAD_RULE, "head_down_r_threshold": head_down_r_threshold,
                                "head_up_r_threshold": head_up_r_threshold}, "status": "insufficient_keypoints"}}
    if raised["raised"]:
        result["state"] = "举手"
        result["head"]["status"] = "skipped_hand_raise"
        return result
    for index in (0, 5, 6):
        if index >= len(points):
            return result
        p = points[index]
        if (len(p) < 3 or not all(math.isfinite(v) for v in p[:3]) or
                p[2] < HEAD_RULE["keypoint_confidence"] or
                not (0 < p[0] < width and 0 < p[1] < height)):
            return result
    shoulder_width = math.hypot(points[5][0]-points[6][0], points[5][1]-points[6][1])
    if shoulder_width < 1:
        result["head"]["status"] = "degenerate_shoulders"
        return result
    ratio = round(((points[5][1]+points[6][1])/2-points[0][1])/shoulder_width, 6)
    result["state"] = "抬头" if ratio > head_up_r_threshold else "低头" if ratio < head_down_r_threshold else "无法判断"
    result["head"].update({"status": "ambiguous_band" if result["state"] == "无法判断" else "estimated", "nose_shoulder_ratio": round(ratio, 6),
                           "shoulder_width": round(shoulder_width, 2)})
    return result
