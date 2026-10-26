"""
CV + DIP Semester | Practical 03: Point Processing
DIP Lecture 01, Slide 16

Point processing mein output pixel ki value sirf usi input pixel se nikalti hai.
Is practical mein brightness, contrast aur image negative apply karenge.
"""

from pathlib import Path

import cv2
import numpy as np

# ROOT directory: lessons/ folder ka parent, jahan datasets/ aur results/ hain.
ROOT = Path(__file__).resolve().parent.parent
input_path = ROOT / "datasets" / "sample.jpg"
output_dir = ROOT / "results" / "point_processing"
output_dir.mkdir(parents=True, exist_ok=True)

# OpenCV image ko BGR order mein uint8 NumPy array ke taur par read karta hai.
image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError(f"Image nahi mili: {input_path}")

# Point processing ki maths samajhne ke liye pixel example:
# f=100, beta=50 -> brightness 150
# f=100, alpha=1.5 -> contrast 150
# f=100 -> negative 155
sample = 100
print("Example input pixel:", sample)
print("Brightness (+50):", min(sample + 50, 255))
print("Contrast (x1.5):", min(round(sample * 1.5), 255))
print("Negative (255-f):", 255 - sample)

# 1) BRIGHTNESS: g = clip(f + beta, 0, 255)
# beta=+50 sab channels ko 50 units brighter karta hai.
# cv2.convertScaleAbs() negative input ko absolute bana deta hai, isliye
# true clipping/brightness ke liye np.clip aur wider signed type use kar rahe hain.
beta = 50
bright = np.clip(image.astype(np.int16) + beta, 0, 255).astype(np.uint8)

# 2) CONTRAST: g = clip(alpha * f, 0, 255)
# alpha > 1 pixel values ko 0 se door push karta hai;
# alpha < 1 values ko 0 ki taraf lata hai. Ye simple scaling hai.
# Note: typical contrast adjustment mid-gray (128) ke around bhi ho sakti hai;
# ye lesson sirf slide ke simple point transformation ko demonstrate karta hai.
alpha = 1.5
contrast = np.clip(image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

# 3) NEGATIVE: g = 255 - f (8-bit image ke liye)
# White black, black white, aur baqi colors complementary ban jate hain.
negative = 255 - image

# Har image ko output directory mein PNG format mein save karo.
outputs = {
    "01_original.png": image,
    "02_brightness_plus_50.png": bright,
    "03_contrast_times_1_5.png": contrast,
    "04_negative.png": negative,
}
for filename, result in outputs.items():
    path = output_dir / filename
    if not cv2.imwrite(str(path), result):
        raise OSError(f"Output save nahi hui: {path}")
    print("Saved:", path.relative_to(ROOT))

# Side-by-side comparison: original + teen processing results.
# np.hstack chaar same-size images ko horizontally join karta hai.
comparison = np.hstack([image, bright, contrast, negative])
comparison_path = output_dir / "05_comparison.png"
cv2.imwrite(str(comparison_path), comparison)
print("Saved:", comparison_path.relative_to(ROOT))
print("Practical 03 complete!")


