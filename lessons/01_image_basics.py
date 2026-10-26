"""
Lesson 01: Basic Image Operations (CV + DIP)

Goal:
    Existing image ko read/display karna, grayscale mein badalna,
    resize, rotate aur crop karna, phir results ko files mein save karna.

Run (project root folder se):
    python lessons/01_image_basics.py

Input:
    datasets/sample.jpg
Outputs:
    results/grayscale.png, resized.png, rotated.png, cropped.png

Note: cv2.imread image ko default BGR channels ke order mein load karta hai.
"""

import cv2  # OpenCV: images read, display, transform aur save karne ki library
from pathlib import Path  # File aur folder paths ko easily handle karne ke liye

# STEP 1: Project ke datasets folder se image read karo.
# imread() JPG file ko NumPy array ki shakal mein memory mein load karta hai.
# Color image ki shape usually (height, width, 3) hoti hai (3 = BGR channels).
image = cv2.imread("datasets/sample.jpg")

# Agar path ghalat ho ya file read na ho, OpenCV None return karta hai.
# Is check se aage ke operations confusing errors nahi denge.
if image is None:
    raise FileNotFoundError("Image not found")

# STEP 2: Saare processed outputs results folder mein save honge.
# exist_ok=True ka matlab folder already ho to dobara banane par error na aaye.
out = Path("results")
out.mkdir(exist_ok=True)

# STEP 3: Original image screen par window mein show karo.
# imshow window open karta hai; waitKey() neeche program ko open rakhta hai.
cv2.imshow("Original", image)

# STEP 4: Color image ko grayscale mein convert karo.
# BGR image mein har pixel ki 3 values hoti hain; grayscale mein 1 brightness value.
# COLOR_BGR2GRAY isliye kyun ke OpenCV ne image BGR order mein read ki hai.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# STEP 5: Image ko width=300, height=300 pixels mein resize karo.
# resize() mein dimensions (width, height) hoti hain, shape mein (height, width).
# Default interpolation pixels ko interpolate karti hai (visual resizing).
resized = cv2.resize(image, (300, 300))

# STEP 6: Image ko 90 degree clockwise rotate karo (right ki taraf).
# Rotation ke baad non-square image ki width/height swap ho sakti hain.
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# STEP 7: Image ka centre wala hissa crop karo.
# shape[:2] se height aur width milti hain, channels ignore ho jate hain.
h, w = image.shape[:2]

# NumPy slicing: image[START_ROW:END_ROW, START_COL:END_COL]
# h//4 = 25% height, 3*h//4 = 75% height; width ke liye bhi wahi.
# Is se image ka central half (width aur height dono ka) select hota hai.
# Python slice ka end index include nahi hota.
cropped = image[h//4:3*h//4, w//4:3*w//4]

# STEP 8: Processed arrays ko files mein save karo.
# str(out / "name.png") pathlib path ko OpenCV-compatible string deta hai.
# imwrite() True/False return karta hai, jo batata hai save successful tha ya nahi.
cv2.imwrite(str(out / "grayscale.png"), gray)
cv2.imwrite(str(out / "resized.png"), resized)
cv2.imwrite(str(out / "rotated.png"), rotated)
cv2.imwrite(str(out / "cropped.png"), cropped)

# STEP 9: Program ko keyboard key press tak wait karwao.
# 0 ka matlab indefinitely wait; phir open ki hui image windows close karo.
cv2.waitKey(0)
cv2.destroyAllWindows()
