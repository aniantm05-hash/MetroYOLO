"""Portable annotation export using Pillow."""

from PIL import ImageDraw, ImageFont

from .domain import Detection


def annotate(image, detections: list[Detection]):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(12, canvas.width // 60))
    for i, detection in enumerate(detections, 1):
        box = (detection.x1, detection.y1, detection.x2, detection.y2)
        draw.rectangle(box, outline="#36E4B4", width=max(2, canvas.width // 300))
        text = f"person {i} | {detection.confidence:.2f}"
        x, y = detection.x1, max(0, detection.y1 - 22)
        bounds = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bounds, fill="#102C35")
        draw.text((x, y), text, font=font, fill="#FFFFFF")
    return canvas
