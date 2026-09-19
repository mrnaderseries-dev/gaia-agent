from __future__ import annotations

"""Generate a large, legible test image containing the number 1729."""

from PIL import Image, ImageDraw, ImageFont

image = Image.new("RGB", (900, 400), "white")
draw = ImageDraw.Draw(image)

font = None
for name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf", "calibri.ttf"):
    try:
        font = ImageFont.truetype(name, 220)
        break
    except OSError:
        continue

if font is None:
    font = ImageFont.load_default()

draw.text((80, 100), "1729", fill="black", font=font)
image.save(r"C:\Users\user\gaia-agent\_audit_data\answer_big.png")
print("written")