
import cv2
from pathlib import Path

# 1. Project ke main folder ka path
ROOT = Path(__file__).resolve().parent.parent

# 2. Original image aur output folder ke paths
image_path = ROOT / "datasets" / "colorful.png"
output_dir = ROOT / "results" / "rgb_channels"
output_dir.mkdir(parents=True, exist_ok=True)

# 3. Image read karo (OpenCV BGR format use karta hai)
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("Image nahi mili!")

# 4. BGR image ko 3 separate channels mein divide karo
blue, green, red = cv2.split(image) 

# 5. Har channel ko separate grayscale image mein save karo
cv2.imwrite(str(output_dir / "01_blue.png"), blue)
cv2.imwrite(str(output_dir / "02_green.png"), green)
cv2.imwrite(str(output_dir / "03_red.png"), red)

# 6. Image aur channel information print karo
print("Original Shape:", image.shape)
print("Blue Channel Shape:", blue.shape)
print("Green Channel Shape:", green.shape)
print("Red Channel Shape:", red.shape)

print("RGB Channels successfully saved!")
