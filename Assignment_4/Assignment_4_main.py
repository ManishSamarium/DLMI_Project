import cv2
import numpy as np
import os

os.makedirs("results", exist_ok=True)

image_path = "nucleus.png"

image = cv2.imread(image_path)
if image is None:
    print("Image not found.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Otsu Thresholding
_, thresh = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Morphological Opening (Noise Removal)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)


# METHOD A: Watershed WITHOUT Marker Control


dist_simple = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

_, sure_fg_simple = cv2.threshold(
    dist_simple, 0.2 * dist_simple.max(), 255, 0
)
sure_fg_simple = np.uint8(sure_fg_simple)

_, markers_simple = cv2.connectedComponents(sure_fg_simple)
markers_simple = markers_simple + 1

image_simple = image.copy()
markers_ws_simple = cv2.watershed(image_simple, markers_simple)

image_simple[markers_ws_simple == -1] = [0, 0, 255]

# METHOD B: Marker-Controlled Watershed


dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

_, sure_fg = cv2.threshold(
    dist, 0.6 * dist.max(), 255, 0
)
sure_fg = np.uint8(sure_fg)

sure_bg = cv2.dilate(opening, kernel, iterations=3)
unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

image_controlled = image.copy()
markers_final = cv2.watershed(image_controlled, markers)

image_controlled[markers_final == -1] = [0, 0, 255]


# Save Results

cv2.imwrite("results/01_original.png", image)
cv2.imwrite("results/02_threshold.png", thresh)
cv2.imwrite("results/03_without_markers.png", image_simple)
cv2.imwrite("results/04_with_markers.png", image_controlled)

