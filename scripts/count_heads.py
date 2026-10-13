"""Count heads in one classroom image using the same backend as the web app."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtdetrv4
import vision


def main() -> None:
    parser = argparse.ArgumentParser(description="RT-DETRv4-S C-RADIOv4 classroom head count")
    parser.add_argument("image", type=Path)
    parser.add_argument("--conf", type=float, default=0.38)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--json", type=Path, help="JSON output path (default: beside the result image)")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"图片不存在：{args.image}")
    if not 0 < args.conf < 1:
        parser.error("conf 必须在 0 和 1 之间")
    started = time.perf_counter()
    # Keep upstream model-construction output out of the JSON stdout stream.
    with contextlib.redirect_stdout(sys.stderr):
        result = vision.detect_image(args.image, weights=rtdetrv4.WEIGHTS_NAME,
                                     conf=args.conf, imgsz=args.imgsz)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    output = args.json or (vision.RESULTS_DIR / result["result"]).with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
