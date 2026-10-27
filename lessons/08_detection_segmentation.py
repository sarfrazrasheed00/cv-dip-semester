
import cv2
import numpy as np
from pathlib import Path

# 1. Project ka output folder
ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "results" / "detection_segmentation"
out.mkdir(parents=True, exist_ok=True)

# 2. Test image create karo
# White background par different colored objects draw karenge.
image = np.full((400, 600, 3), 255, dtype=np.uint8)

# Red circle (BGR mein red = 0, 0, 255)
cv2.circle(image, (150, 160), 75, (0, 0, 255), -1)

# Green rectangle
cv2.rectangle(image, (320, 90), (500, 230), (0, 200, 0), -1)

# Chhota red circle
cv2.circle(image, (400, 320), 35, (0, 0, 255), -1)

# 3. BGR ko HSV color space mein convert karo.
# HSV mein colors ki ranges select karna convenient hota hai.
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 4. Red color ki segmentation mask banao.
# HSV hue circular hota hai, isliye red ki 2 ranges hain.
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([179, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

# Dono ranges combine karke final binary mask
mask = cv2.bitwise_or(mask1, mask2)

# 5. Segmentation: sirf red objects ke pixels retain karo.
segmented = cv2.bitwise_and(image, image, mask=mask)

# 6. Object Detection: mask se object contours nikalo.
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

detected = image.copy()
count = 0

for contour in contours:
    # Bohat chhote regions ignore karo.
    if cv2.contourArea(contour) < 100:
        continue

    # Bounding box calculate karo.
    x, y, w, h = cv2.boundingRect(contour)

    # Detected object ke around black rectangle draw karo.
    cv2.rectangle(
        detected, (x, y), (x + w, y + h),
        (0, 0, 0), 3
    )

    count += 1

print("Total red objects detected:", count)

# 7. Outputs save karo.
cv2.imwrite(str(out / "01_original.png"), image)
cv2.imwrite(str(out / "02_segmentation_mask.png"), mask)
cv2.imwrite(str(out / "03_segmented_objects.png"), segmented)
cv2.imwrite(str(out / "04_object_detection.png"), detected)

print("Lesson 08 results saved!")
