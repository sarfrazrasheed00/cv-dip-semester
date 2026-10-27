
import cv2
import numpy as np
from pathlib import Path

# Project ka main folder
ROOT = Path(__file__).resolve().parent.parent

# Input image
image_path = ROOT / "datasets" / "sample.jpg"

# Output folder
output_dir = ROOT / "results" / "convolution"
output_dir.mkdir(parents=True, exist_ok=True)

# Original image read karna
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("Image nahi mili")

# Grayscale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Average Blur: 3x3 neighborhood ka average
average_kernel = np.ones((3, 3), dtype=np.float32) / 9
average_blur = cv2.filter2D(gray, -1, average_kernel)

# 2. Gaussian Blur: center pixels ko zyada weight
gaussian_blur = cv2.GaussianBlur(gray, (3, 3), 0)

# 3. Sharpen: center intensity ko emphasize karna
sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

# Results save karna
cv2.imwrite(str(output_dir / "01_original.png"), gray)
cv2.imwrite(str(output_dir / "02_average_blur.png"), average_blur)
cv2.imwrite(str(output_dir / "03_gaussian_blur.png"), gaussian_blur)
cv2.imwrite(str(output_dir / "04_sharpened.png"), sharpened)

print("Convolution results successfully saved!")
