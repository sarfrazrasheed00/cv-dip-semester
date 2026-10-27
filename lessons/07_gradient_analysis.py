
import cv2
import numpy as np
from pathlib import Path

# 1. Project ke paths
ROOT = Path(__file__).resolve().parent.parent
image_path = ROOT / "datasets" / "sample.jpg"

output_dir = ROOT / "results" / "gradient_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Image grayscale mein read karo
# Sobel ko intensity image par apply karenge.
gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise FileNotFoundError(f"Image nahi mili: {image_path}")

# 3. Image statistics
# Har pixel ki brightness 0 se 255 ke darmiyan hai.
mean_value = np.mean(gray)
minimum = np.min(gray)
maximum = np.max(gray)
intensity_range = int(maximum) - int(minimum)

print("Shape:", gray.shape)
print("Mean:", mean_value)
print("Min:", minimum)
print("Max:", maximum)
print("Intensity Range:", intensity_range)

# 4. Sobel X aur Sobel Y
# Gx = left-right direction ka intensity change.
# Gy = top-bottom direction ka intensity change.
# CV_64F negative gradients ko bhi preserve karta hai.
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# 5. Gradient Magnitude
# Formula: G = sqrt(Gx^2 + Gy^2)
# Ye overall edge strength calculate karta hai.
magnitude = cv2.magnitude(gx, gy)

# Visualization ke liye magnitude ko 0-255 mein normalize karo.
magnitude_view = cv2.normalize(
    magnitude, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# 6. Gradient Direction
# atan2(Gy, Gx) se angle radians mein milta hai.
# degrees() angle ko degrees mein convert karta hai.
direction = np.degrees(np.arctan2(gy, gx))

# Direction ko 0-360 degree range mein convert karo.
direction = (direction + 360) % 360

# Sirf visualization ke liye angle ko grayscale mein map karo.
direction_view = np.uint8(direction * 255 / 360)

# 7. Threshold percentage
# Check karo kitne pixels intensity 100 se zyada rakhte hain.
threshold = 100
bright_pixels = np.count_nonzero(gray > threshold)
total_pixels = gray.size

percentage = (bright_pixels / total_pixels) * 100

print("Total Pixels:", total_pixels)
print("Pixels Above 100:", bright_pixels)
print("Percentage Above 100:", round(percentage, 2), "%")

# 8. Numerical example
# Gradient X = 30, Gradient Y = 40
example_gx = 30
example_gy = 40

example_magnitude = np.sqrt(example_gx**2 + example_gy**2)
example_direction = np.degrees(
    np.arctan2(example_gy, example_gx)
)

print("Example Magnitude:", example_magnitude)
print("Example Direction:", round(example_direction, 2))

# 9. Processed images save karo.
cv2.imwrite(str(output_dir / "01_grayscale.png"), gray)
cv2.imwrite(
    str(output_dir / "02_sobel_x.png"),
    cv2.convertScaleAbs(gx)
)
cv2.imwrite(
    str(output_dir / "03_sobel_y.png"),
    cv2.convertScaleAbs(gy)
)
cv2.imwrite(
    str(output_dir / "04_gradient_magnitude.png"),
    magnitude_view
)
cv2.imwrite(
    str(output_dir / "05_gradient_direction.png"),
    direction_view
)

print("Lesson 07 outputs successfully saved!")
