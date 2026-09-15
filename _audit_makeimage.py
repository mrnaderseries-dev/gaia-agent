from __future__ import annotations

"""Generate a deterministic test image containing the number 1729."""

from PIL import Image, ImageDraw

image = Image.new("RGB", (640, 240), "white")
draw = ImageDraw.Draw(image)
draw.text((40, 90), "1729", fill="black")
image.save(r"C:\Users\user\gaia-agent\_audit_data\answer.png")
print("written")