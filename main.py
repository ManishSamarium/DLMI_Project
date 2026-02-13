import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
from tqdm import tqdm

IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

def dice_score(pred, gt):
    pred = pred > 0
    gt = gt > 0
    intersection = np.logical_and(pred, gt).sum()
    return (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)

def jaccard_score(pred, gt):
    pred = pred > 0
    gt = gt > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)

dice_otsu = []
dice_sauvola = []
jac_otsu = []
jac_sauvola = []

files = os.listdir(IMAGE_DIR)

for file in tqdm(files):

    img_path = os.path.join(IMAGE_DIR, file)
    mask_path = os.path.join(MASK_DIR, file)

    if not os.path.exists(mask_path):
        continue

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure mask is binary
    _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

    # Smooth image
    image = cv2.GaussianBlur(image, (5,5), 0)

    # OTSU
    _, otsu_mask = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # SAUVOLA
    window_size = 25
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)
    sauvola_mask = image > thresh_sauvola
    sauvola_mask = sauvola_mask.astype(np.uint8) * 255

    # Morphology cleanup
    kernel = np.ones((3,3), np.uint8)
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel)
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)

    sauvola_mask = cv2.morphologyEx(sauvola_mask, cv2.MORPH_OPEN, kernel)
    sauvola_mask = cv2.morphologyEx(sauvola_mask, cv2.MORPH_CLOSE, kernel)

    # Metrics
    dice_otsu.append(dice_score(otsu_mask, gt_mask))
    dice_sauvola.append(dice_score(sauvola_mask, gt_mask))

    jac_otsu.append(jaccard_score(otsu_mask, gt_mask))
    jac_sauvola.append(jaccard_score(sauvola_mask, gt_mask))

print("\n===== FINAL RESULTS =====")
print("Average Dice (Otsu):", np.mean(dice_otsu))
print("Average Dice (Sauvola):", np.mean(dice_sauvola))
print("Average Jaccard (Otsu):", np.mean(jac_otsu))
print("Average Jaccard (Sauvola):", np.mean(jac_sauvola))




plt.figure(figsize=(12,4))
plt.subplot(1,4,1); plt.imshow(image, cmap='gray'); plt.title("Original")
plt.subplot(1,4,2); plt.imshow(gt_mask, cmap='gray'); plt.title("Ground Truth")
plt.subplot(1,4,3); plt.imshow(otsu_mask, cmap='gray'); plt.title("Otsu")
plt.subplot(1,4,4); plt.imshow(sauvola_mask, cmap='gray'); plt.title("Sauvola")
plt.show()
