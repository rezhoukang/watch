"""Estimate visible person context from a Head box in original-image pixels."""
import math
from dataclasses import asdict, dataclass

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Expansion:
    side: float = 0.85
    top: float = 0.0
    below: float = 1.3

    def __post_init__(self):
        for name, value, maximum in (("side", self.side, 5), ("top", self.top, 3),
                                      ("below", self.below, 12)):
            if not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"{name} 超出有效范围")

    def to_dict(self) -> dict:
        return asdict(self)


def validate_box(box: list, width: int, height: int) -> list[int]:
    if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
        raise ValueError("区域需要四个有效坐标 (x1, y1, x2, y2)")
    x1, y1, x2, y2 = box
    result = [max(0, math.floor(x1)), max(0, math.floor(y1)),
              min(width, math.ceil(x2)), min(height, math.ceil(y2))]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("区域为空或位于图片之外")
    return result


def estimate(head: list, width: int, height: int, expansion: Expansion) -> list[int]:
    x1, y1, x2, y2 = validate_box(head, width, height)
    w, h = x2-x1, y2-y1
    return validate_box([x1-w*expansion.side, y1-h*expansion.top,
                         x2+w*expansion.side, y2+h*expansion.below], width, height)


def render_crop(image: Image.Image, region: list, head: list) -> tuple[Image.Image, Image.Image]:
    """Keep a clean rectangular crop and mark only the target head for VLM."""
    box = validate_box(region, *image.size)
    raw = image.crop((box[0], box[1], box[2], box[3]))
    marked = raw.copy()
    drawing = ImageDraw.Draw(marked)
    x1, y1, x2, y2 = head
    drawing.rectangle((x1-box[0], y1-box[1], x2-box[0], y2-box[1]),
                      outline="#ff365e", width=max(1, min(image.size)//650))
    return raw, marked
