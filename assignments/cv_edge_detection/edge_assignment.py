import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Q3 parameters
# -----------------------------
SIGMA = 1.0
KERNEL_SIZE = 7
T_LOW = 0.10
T_HIGH = 0.30

folder = Path(__file__).resolve().parent
image_path = folder / "test_image.png"
figures = folder / "figures"
figures.mkdir(exist_ok=True)

# -----------------------------
# Read image and convert to gray
# -----------------------------
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("test_image.png not found")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float64) / 255.0


# -----------------------------
# Q3(a) Gaussian smoothing
# -----------------------------
# Gaussian kernel formula se khud banaya hai.
# GaussianBlur() use nahi kiya.

def gaussian_kernel(size, sigma):
    half = size // 2
    coords = np.arange(-half, half + 1)
    x, y = np.meshgrid(coords, coords)

    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(
        -(x**2 + y**2) / (2 * sigma**2)
    )

    return kernel / kernel.sum()


kernel = gaussian_kernel(KERNEL_SIZE, SIGMA)
smoothed = cv2.filter2D(gray, -1, kernel)

print("Gaussian kernel sum:", kernel.sum())

# Q3(a) figure
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].imshow(gray, cmap="gray")
ax[0].set_title("Original Grayscale [0,1]")
ax[0].axis("off")

ax[1].imshow(smoothed, cmap="gray")
ax[1].set_title(f"Gaussian Smoothed\nσ={SIGMA}, k={KERNEL_SIZE}")
ax[1].axis("off")

plt.tight_layout()
plt.savefig(figures / "Q3(a)_Gaussian_Smoothing.png", dpi=300)
plt.show()


# -----------------------------
# Q3(b) Sobel gradients
# -----------------------------
sx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)

sy = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float64)

gx = cv2.filter2D(smoothed, cv2.CV_64F, sx)
gy = cv2.filter2D(smoothed, cv2.CV_64F, sy)

# Overall edge strength and direction
magnitude = np.sqrt(gx**2 + gy**2)
direction = np.degrees(np.arctan2(gy, gx))

# Thresholding ke liye magnitude ko [0,1] normalize kiya.
max_mag = magnitude.max()
mag_norm = magnitude / max_mag if max_mag > 0 else magnitude

# Q3(b) figure
fig, ax = plt.subplots(2, 2, figsize=(12, 9))

ax[0, 0].imshow(gx, cmap="gray")
ax[0, 0].set_title("Sobel Gx")

ax[0, 1].imshow(gy, cmap="gray")
ax[0, 1].set_title("Sobel Gy")

ax[1, 0].imshow(mag_norm, cmap="gray")
ax[1, 0].set_title("Normalized Gradient Magnitude")

direction_plot = ax[1, 1].imshow(direction, cmap="hsv")
ax[1, 1].set_title("Gradient Direction")
fig.colorbar(direction_plot, ax=ax[1, 1], label="Degrees")

for axis in ax.ravel():
    axis.axis("off")

plt.tight_layout()
plt.savefig(figures / "Q3(b)_Gradient_Results.png", dpi=300)
plt.show()


# -----------------------------
# Q3(c) Double threshold + hysteresis
# -----------------------------
strong = mag_norm >= T_HIGH
weak = (mag_norm >= T_LOW) & (mag_norm < T_HIGH)

# Strong edge se connected weak pixels ko final edge mein rakho.
final_edges = np.zeros_like(strong, dtype=np.uint8)
final_edges[strong] = 255

stack = list(zip(*np.where(strong)))
visited = np.zeros_like(strong, dtype=bool)

neighbours = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
]

rows, cols = strong.shape

while stack:
    r, c = stack.pop()

    for dr, dc in neighbours:
        nr, nc = r + dr, c + dc

        if 0 <= nr < rows and 0 <= nc < cols:
            if weak[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                final_edges[nr, nc] = 255
                stack.append((nr, nc))

# Q3(c) figure
fig, ax = plt.subplots(2, 2, figsize=(12, 9))

ax[0, 0].imshow(mag_norm, cmap="gray")
ax[0, 0].set_title("Normalized Magnitude")

ax[0, 1].imshow(strong, cmap="gray")
ax[0, 1].set_title(f"Strong Edges (>= {T_HIGH})")

ax[1, 0].imshow(weak, cmap="gray")
ax[1, 0].set_title(f"Weak Edges ({T_LOW} to {T_HIGH})")

ax[1, 1].imshow(final_edges, cmap="gray")
ax[1, 1].set_title("Final Edges After Hysteresis")

for axis in ax.ravel():
    axis.axis("off")

plt.tight_layout()
plt.savefig(figures / "Q3(c)_Threshold_Hysteresis.png", dpi=300)
plt.show()


# -----------------------------
# Q3(d) Built-in comparison
# -----------------------------
# Built-in Sobel
sobel_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

if sobel_mag.max() > 0:
    sobel_mag = sobel_mag / sobel_mag.max()

# Built-in Canny ko same smoothed image dete hain.
smoothed_8bit = np.clip(smoothed * 255, 0, 255).astype(np.uint8)

# Hamare normalized thresholds ko Sobel gradient scale ke qareeb convert kiya.
canny_low = T_LOW * max_mag * 255
canny_high = T_HIGH * max_mag * 255

canny = cv2.Canny(
    smoothed_8bit,
    canny_low,
    canny_high,
    L2gradient=True
)

original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(2, 2, figsize=(14, 9))

ax[0, 0].imshow(original_rgb)
ax[0, 0].set_title("Original Image")

ax[0, 1].imshow(sobel_mag, cmap="gray")
ax[0, 1].set_title("Built-in Sobel")

ax[1, 0].imshow(final_edges, cmap="gray")
ax[1, 0].set_title("Our Pipeline")

ax[1, 1].imshow(canny, cmap="gray")
ax[1, 1].set_title("Built-in Canny")

for axis in ax.ravel():
    axis.axis("off")

plt.tight_layout()
plt.savefig(figures / "Q3(d)_Comparison.png", dpi=300)
plt.show()

print("Q3 completed")
print("Our edge pixels:", np.count_nonzero(final_edges))
print("Built-in Canny edge pixels:", np.count_nonzero(canny))

"""
First I converted the image to grayscale and normalized it to [0,1]. Then I created the Gaussian kernel manually and smoothed the image. After that I applied Sobel X and Y filters to calculate gradient magnitude and direction. I used double thresholding to classify strong and weak edges and applied 8-connected hysteresis to preserve connected weak edges. Finally, I compared my result with built-in Sobel and OpenCV Canny.

Gaussian → noise smooth karta hai.  
Sobel X/Y → intensity changes find karta hai.  
Magnitude → edge strength.  
Direction → gradient direction.  
Double threshold → strong/weak edges separate karta hai.  
Hysteresis → strong se connected weak edges preserve karta hai.

"""
