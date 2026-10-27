"""
LESSON 06: IMAGE SAMPLING AUR QUANTIZATION

Sampling: pixels ki spatial quantity/resolution kam karna.
Quantization: har pixel ki possible intensity values/levels kam karna.

Input: datasets/sample.jpg
Output: results/sampling_quantization/
"""

from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "datasets" / "sample.jpg"
OUTPUT_DIR = ROOT / "results" / "sampling_quantization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Grayscale se shuru kar rahe hain, kyun ke 8-bit intensity 0 se 255
# tak aur quantization ke levels samajhna is mein sab se asaan hai.
image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Image nahi mili: {IMAGE_PATH}")

height, width = image.shape  # Grayscale image: (rows, columns)
print(f"Original image: {width} x {height} pixels")
cv2.imwrite(str(OUTPUT_DIR / "01_original_grayscale.png"), image)

# --------------------------------------------------------------------
# PART A: SAMPLING — resolution ko reduce karna
# --------------------------------------------------------------------
# 612x612 original ko 64x64 aur 16x16 mein downsample karenge.
# INTER_AREA downsampling ke liye achha default hai: chhote result mein
# region ka overall appearance preserve karne mein help karta hai.
# Uske baad INTER_NEAREST se wapas original display size par enlarge
# karenge. Nearest-neighbor blocks ko clear dikhata hai (pixelation).
# Important: enlarged image mein original detail wapas NAHI aati.
for size in (64, 16):
    sampled = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    enlarged = cv2.resize(
        sampled, (width, height), interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite(str(OUTPUT_DIR / f"02_sampling_{size}x{size}_actual.png"), sampled)
    cv2.imwrite(str(OUTPUT_DIR / f"03_sampling_{size}x{size}_enlarged.png"), enlarged)
    print(f"Sampling: {width}x{height} -> {size}x{size}")

# --------------------------------------------------------------------
# PART B: QUANTIZATION — brightness ke possible levels kam karna
# --------------------------------------------------------------------
# Original grayscale 8-bit: 2^8 = 256 values (0...255).
# 4-bit = 16 levels, 2-bit = 4 levels, 1-bit = 2 levels.
# Hum output ko PNG uint8 hi save karte hain taa-ke images asaani se
# dekhi ja saken. Is liye ye actual 1/2/4-bit file encoding NAHI hai;
# sirf quantization ke available intensity levels simulate karta hai.
# Mapping: n levels ko 0...255 range par equally spaced rakhna.

def quantize_grayscale(gray: np.ndarray, bits: int) -> np.ndarray:
    """Nearest evenly-spaced level par 8-bit intensities map karta hai."""
    levels = 2 ** bits
    # float32 use karte hain taa-ke intermediate rounding safe rahe.
    scaled = gray.astype(np.float32) * (levels - 1) / 255.0
    quantized = np.rint(scaled) * 255.0 / (levels - 1)
    return np.clip(np.rint(quantized), 0, 255).astype(np.uint8)

quantized_images = {}
for bits in (8, 4, 2, 1):
    result = quantize_grayscale(image, bits)
    quantized_images[bits] = result
    cv2.imwrite(str(OUTPUT_DIR / f"04_quantization_{bits}bit.png"), result)
    distinct_values = np.unique(result)
    print(f"Quantization {bits}-bit: max {2**bits} possible levels; "
          f"image mein {len(distinct_values)} levels use hue")

# --------------------------------------------------------------------
# PART C: COMPARISON — saare results ek saath
# --------------------------------------------------------------------
# Har image ka size same hai, is liye hstack/vstack se grid ban sakti hai.
# Sampling comparison (original, 64x64 enlarged, 16x16 enlarged)
small_64 = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
small_16 = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
display_64 = cv2.resize(small_64, (width, height), interpolation=cv2.INTER_NEAREST)
display_16 = cv2.resize(small_16, (width, height), interpolation=cv2.INTER_NEAREST)
sampling_comparison = np.hstack([image, display_64, display_16])
cv2.imwrite(str(OUTPUT_DIR / "05_sampling_comparison.png"), sampling_comparison)

# Quantization comparison: original/8-bit, 4-bit, 2-bit, 1-bit
quantization_comparison = np.hstack([
    quantized_images[8], quantized_images[4],
    quantized_images[2], quantized_images[1]
])
cv2.imwrite(str(OUTPUT_DIR / "06_quantization_comparison.png"), quantization_comparison)

# Ek pixel ki numerical example se samjho:
example = np.array([[180]], dtype=np.uint8)
for bits in (8, 4, 2, 1):
    print(f"Pixel 180 ko {bits}-bit mein: {int(quantize_grayscale(example, bits)[0, 0])}")

print(f"\nAll outputs saved: {OUTPUT_DIR}")


import cv2

image = cv2.imread("results/convolution/01_original.png", cv2.IMREAD_UNCHANGED)

print("Shape:", image.shape)
print("Data Type:", image.dtype)
print("Bits per Channel:", image.dtype.itemsize * 8)