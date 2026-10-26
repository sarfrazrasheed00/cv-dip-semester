"""
CV Lecture 02: Image Properties, Sobel, Canny aur Thresholding

Goal:
    Image ko grayscale banana, brightness statistics calculate karna,
    edges detect karna aur threshold se binary image banana.

Run (project root folder se):
    python lessons/02_cv_lecture_02.py

Input:
    datasets/sample.jpg
Outputs:
    results/cv_lecture_02/grayscale.png
    results/cv_lecture_02/sobel_x.png, sobel_y.png
    results/cv_lecture_02/canny.png, threshold.png
"""

import cv2  # OpenCV: image processing aur edge detection ke functions
from pathlib import Path  # Files/folders ke platform-independent paths

# STEP 1: Sample image ko memory mein NumPy array ke taur par load karo.
# OpenCV default BGR (Blue, Green, Red) channel order use karta hai.
image = cv2.imread("datasets/sample.jpg")
print("Shape:", image.shape)

# Image missing ho ya read na ho to program meaningful error dega.
if image is None:
    raise FileNotFoundError("Image not found")

# STEP 2: Lecture 02 ke outputs ke liye alag folder banao.
# parents=True zarurat par parent 'results' folder bhi bana dega.
out = Path("results/cv_lecture_02")
out.mkdir(parents=True, exist_ok=True)

# STEP 3: BGR color image ko grayscale mein badlo.
# Color pixel ki 3 channel values ki jagah ab sirf 1 intensity value hogi.
# Standard 8-bit grayscale mein intensity range 0 (black) se 255 (white) hai.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# STEP 4: Image ki numerical properties dekho.
# Shape = (height, width, channels) color image ke liye.
# Mean = sab grayscale pixels ka average (overall brightness ka rough measure).
# Min aur Max = image mein actual darkest aur brightest pixel values.
print("Mean:", gray.mean())
print("Min:", gray.min())
print("Max:", gray.max())

# STEP 5: Sobel edge detection.
# Edge woh region hai jahan image intensity tez change hoti hai.
# Sobel neighbourhood (default 3x3 kernel) se intensity ka gradient estimate karta hai.
# cv2.CV_64F output floating-point rakhta hai, negative gradients bhi preserve hotay hain.
# dx=1, dy=0 => left-right direction mein change (vertical boundaries par response).
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
# dx=0, dy=1 => up-down direction mein change (horizontal boundaries par response).
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

# STEP 6: Canny edge detector.
# Canny multiple stages use karta hai: noise smoothing, gradients,
# non-maximum suppression, double threshold aur weak/strong edge linking.
# 100 = lower threshold, 200 = upper threshold (gradient strength units).
# Ye fixed example values hain; different images par tune karna par sakta hai.
canny = cv2.Canny(gray, 100, 200)

# STEP 7: Binary thresholding.
# Threshold 100: >= 100 intensity ko white (255), baqi ko black (0) banao.
# cv2.threshold() do cheezen return karta hai: used threshold aur output image.
# '_' ka matlab pehli returned value abhi use nahi kar rahe.
_, binary = cv2.threshold(
    gray, 100, 255, cv2.THRESH_BINARY
)

# STEP 8: Processing outputs save karo.
# Sobel results mein negative/float values ho sakti hain, isliye
# convertScaleAbs() absolute gradient le kar 8-bit display image banata hai.
# Note: absolute conversion gradient ka positive/negative direction lose karta hai.
cv2.imwrite(str(out / "grayscale.png"), gray)
cv2.imwrite(
    str(out / "sobel_x.png"),
    cv2.convertScaleAbs(sobel_x)
)
cv2.imwrite(
    str(out / "sobel_y.png"),
    cv2.convertScaleAbs(sobel_y)
)
cv2.imwrite(str(out / "canny.png"), canny)
cv2.imwrite(str(out / "threshold.png"), binary)

# STEP 9: Completion message; output folder mein generated images check karo.
print("All outputs saved!")
