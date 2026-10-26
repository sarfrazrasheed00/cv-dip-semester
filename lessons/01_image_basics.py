import cv2
from pathlib import Path

image = cv2.imread("datasets/sample.jpg")

if image is None:
    raise FileNotFoundError("Image not found")

out = Path("results")
out.mkdir(exist_ok=True)

# 1. Read & Display
cv2.imshow("Original", image)

# 2. Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Resize to 300x300
resized = cv2.resize(image, (300, 300))

# 4. Rotate 90 degrees clockwise
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# 5. Crop center portion
h, w = image.shape[:2]
cropped = image[h//4:3*h//4, w//4:3*w//4]

# Save results
cv2.imwrite(str(out / "grayscale.png"), gray)
cv2.imwrite(str(out / "resized.png"), resized)
cv2.imwrite(str(out / "rotated.png"), rotated)
cv2.imwrite(str(out / "cropped.png"), cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()