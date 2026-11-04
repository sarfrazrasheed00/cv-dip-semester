import cv2
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

image = cv2.imread(str(ROOT / "datasets" / "road.jpg"))

if image is None:
    raise FileNotFoundError("road.jpg not found")

# Color image -> grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thora noise/texture smooth karo
blurred = cv2.GaussianBlur(gray, (5, 5), 1)

cv2.imwrite(str(ROOT / "results" / "road_gray.png"), gray)
cv2.imwrite(str(ROOT / "results" / "road_blurred.png"), blurred)

# Sobel X: left-right intensity change detect karta hai
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)

# Sobel Y: top-bottom intensity change detect karta hai
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Negative values ko positive display form mein convert karo
sobel_x_view = cv2.convertScaleAbs(sobel_x)
sobel_y_view = cv2.convertScaleAbs(sobel_y)

cv2.imwrite(str(ROOT / "results" / "road_sobel_x.png"), sobel_x_view)
cv2.imwrite(str(ROOT / "results" / "road_sobel_y.png"), sobel_y_view)

magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude_view = cv2.convertScaleAbs(magnitude)

cv2.imwrite(
    str(ROOT / "results" / "road_gradient_magnitude.png"),
    magnitude_view
)

# Gradient magnitude ko 0-255 range mein normalize karo
magnitude_norm = cv2.normalize(
    magnitude, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Sirf strong edges rakho
# Edge strength < 80  → Black (ignore)
# Edge strength >= 80 → White (keep)

# threshold_value, thresholded = cv2.threshold(
#     magnitude_norm, 80, 255, cv2.THRESH_BINARY
# )

# print(threshold_value)   # 80.0

# cv2.imwrite(
#     str(ROOT / "results" / "road_thresholded.png"),
#     thresholded
# )


# Simple logic:
# - >=100 → strong edge, direct keep.
# - 40–99 → weak edge.
# - Weak edge sirf tab keep hogi jab strong edge se connected ho.
# - <40 → remove.

T_LOW = 40
T_HIGH = 100

strong = magnitude_norm >= T_HIGH
weak = (magnitude_norm >= T_LOW) & (magnitude_norm < T_HIGH)

final_edges = np.zeros_like(magnitude_norm, dtype=np.uint8)
final_edges[strong] = 255

stack = list(zip(*np.where(strong)))
visited = np.zeros_like(strong, dtype=bool)

neighbors = [
    (-1,-1), (-1,0), (-1,1),
    ( 0,-1),         ( 0,1),
    ( 1,-1), ( 1,0), ( 1,1)
]

rows, cols = magnitude_norm.shape

while stack:
    r, c = stack.pop()

    for dr, dc in neighbors:
        nr, nc = r + dr, c + dc

        if 0 <= nr < rows and 0 <= nc < cols:
            if weak[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                final_edges[nr, nc] = 255
                stack.append((nr, nc))

cv2.imwrite(
    str(ROOT / "results" / "road_hysteresis.png"),
    final_edges
)

canny_edges = cv2.Canny(blurred, 50, 150)

cv2.imwrite(
    str(ROOT / "results" / "road_canny.png"),
    canny_edges
)

mask = np.zeros_like(canny_edges)

height, width = canny_edges.shape

polygon = np.array([[
    (0, height),
    (width, height),
    (int(width * 0.65), int(height * 0.30)),
    (int(width * 0.35), int(height * 0.30))
]], dtype=np.int32)

cv2.fillPoly(mask, polygon, 255)

road_edges = cv2.bitwise_and(canny_edges, mask)

cv2.imwrite(
    str(ROOT / "results" / "road_roi.png"),
    road_edges
)


print("Done")