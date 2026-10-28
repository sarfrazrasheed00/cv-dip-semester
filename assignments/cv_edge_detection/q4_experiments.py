import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Q4(a) Parameters
# -----------------------------
T_LOW = 0.10
T_HIGH = 0.30
SIGMA_VALUES = [0.5, 1, 2, 4]

folder = Path(__file__).resolve().parent
image_path = folder / "test_image.png"
figures = folder / "figures"
figures.mkdir(exist_ok=True)

# -----------------------------
# Read and prepare image
# -----------------------------
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("test_image.png not found")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float64) / 255.0


# -----------------------------
# Gaussian kernel
# Same method used in Q3
# -----------------------------
def gaussian_kernel(size, sigma):
    half = size // 2
    coords = np.arange(-half, half + 1)
    x, y = np.meshgrid(coords, coords)

    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(
        -(x**2 + y**2) / (2 * sigma**2)
    )

    return kernel / kernel.sum()


# Sobel kernels
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


# -----------------------------
# Our Q3 pipeline
# -----------------------------
def detect_edges(gray_image, sigma, t_low=T_LOW, t_high=T_HIGH):

    # Required kernel-size formula:
    # k = 2 * ceil(3*sigma) + 1
    k = 2 * int(np.ceil(3 * sigma)) + 1

    kernel = gaussian_kernel(k, sigma)
    smoothed = cv2.filter2D(gray_image, -1, kernel)

    gx = cv2.filter2D(smoothed, cv2.CV_64F, sx)
    gy = cv2.filter2D(smoothed, cv2.CV_64F, sy)

    magnitude = np.sqrt(gx**2 + gy**2)

    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    # Double threshold
    strong = magnitude >= t_high
    weak = (magnitude >= t_low) & (magnitude < t_high)

    # Hysteresis
    edges = np.zeros_like(strong, dtype=np.uint8)
    edges[strong] = 255

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
                    edges[nr, nc] = 255
                    stack.append((nr, nc))

    return edges, k


# -----------------------------
# Test all sigma values
# -----------------------------
results = []

for sigma in SIGMA_VALUES:
    edges, kernel_size = detect_edges(gray, sigma)
    edge_count = np.count_nonzero(edges)

    results.append(
        (sigma, kernel_size, edges, edge_count)
    )

    print(
        f"Sigma = {sigma}, "
        f"Kernel = {kernel_size}x{kernel_size}, "
        f"Edge Pixels = {edge_count}"
    )


# -----------------------------
# Q4(a) 2x2 comparison figure
# -----------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

for axis, result in zip(ax.ravel(), results):

    sigma, kernel_size, edges, edge_count = result

    axis.imshow(edges, cmap="gray")

    axis.set_title(
        f"σ = {sigma}, k = {kernel_size}x{kernel_size}\n"
        f"Edge Pixels = {edge_count}"
    )

    axis.axis("off")


plt.tight_layout()

plt.savefig(
    figures / "Q4(a)_Sigma_Comparison.png",
    dpi=300
)

plt.show()

print("\nQ4(a) completed.")


# ============================================================
# Q4(b) - Effect of Thresholds
# ============================================================

BEST_SIGMA = 2

threshold_pairs = [
    (0.05, 0.10),
    (0.10, 0.30),
    (0.30, 0.60)
]

q4b_results = []

for low, high in threshold_pairs:

    edges, kernel_size = detect_edges(
        gray,
        BEST_SIGMA,
        low,
        high
    )

    edge_count = np.count_nonzero(edges)

    q4b_results.append(
        (low, high, edges, edge_count)
    )

    print(
        f"T_low = {low}, "
        f"T_high = {high}, "
        f"Edge Pixels = {edge_count}"
    )


# -----------------------------
# Q4(b) comparison figure
# -----------------------------
fig, ax = plt.subplots(
    1,
    3,
    figsize=(17, 6)
)

for axis, result in zip(
    ax,
    q4b_results
):

    low, high, edges, edge_count = result

    axis.imshow(
        edges,
        cmap="gray"
    )

    axis.set_title(
        f"T_low={low}, T_high={high}\n"
        f"Edge Pixels={edge_count}"
    )

    axis.axis("off")


plt.tight_layout()

plt.savefig(
    figures / "Q4(b)_Threshold_Comparison.png",
    dpi=300
)

plt.show()

print("\nQ4(b) completed.")