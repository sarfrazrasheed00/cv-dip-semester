"""Lesson 01: Digital images as numerical arrays (CV + DIP)."""
from pathlib import Path
import cv2
import numpy as np

# A 3x3 grayscale image: each value is one pixel intensity.
image = np.array([[0, 128, 255], [50, 100, 200], [25, 150, 230]], dtype=np.uint8)
print('Image matrix:\n', image)
print('Shape:', image.shape)
print('Pixel (row=1, col=1):', image[1, 1])
print('Bit depth per pixel:', image.dtype.itemsize * 8)
print('Raw bytes:', image.nbytes)

out = Path('results')
out.mkdir(exist_ok=True)
# Nearest-neighbour upscale is only for visualization, preserving discrete pixel values.
preview = cv2.resize(image, (300, 300), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(out / 'lesson_01_pixels.png'), preview)
print('Saved:', out / 'lesson_01_pixels.png')
