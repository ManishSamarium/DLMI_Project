import os
import cv2
import numpy as np
from skimage.filters import threshold_niblack, threshold_sauvola


# PATHS

image_folder = "training/images/"
gt_folder = "training/1st_manual/"
mask_folder = "training/mask/"


# PARAMETERS (tuned for DRIVE)

window_size = 25
k_niblack = 0.2
k_sauvola = 0.1


# METRIC STORAGE

sens_niblack_list = []
sens_sauvola_list = []

dice_niblack_list = []
dice_sauvola_list = []

jacc_niblack_list = []
jacc_sauvola_list = []

saved_sample = False



# METRIC FUNCTIONS

def compute_sensitivity(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    return tp / (tp + fn) if (tp + fn) != 0 else 0


def compute_dice(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return (2 * intersection) / total if total != 0 else 0


def compute_jaccard(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union != 0 else 0


# MAIN LOOP

for file in os.listdir(image_folder):
    if file.endswith(".tif"):

        img_path = os.path.join(image_folder, file)
        gt_path = os.path.join(gt_folder, file.replace("_training.tif", "_manual1.gif"))
        mask_path = os.path.join(mask_folder, file.replace(".tif", "_mask.gif"))

        image = cv2.imread(img_path)
        gt = cv2.imread(gt_path, 0)
        mask = cv2.imread(mask_path, 0)

        if image is None or gt is None or mask is None:
            continue

        # -------- PREPROCESSING --------
        green = image[:, :, 1]

        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        green = clahe.apply(green)

        green_blur = cv2.GaussianBlur(green, (5, 5), 0)

        gt = gt > 0
        mask = mask > 0

        # NIBLACK 
        thresh_n = threshold_niblack(green_blur, window_size, k=k_niblack)

        # vessels are dark in DRIVE
        binary_niblack = green_blur < thresh_n
        binary_niblack = np.logical_and(binary_niblack, mask)

        sens_niblack_list.append(compute_sensitivity(binary_niblack, gt))
        dice_niblack_list.append(compute_dice(binary_niblack, gt))
        jacc_niblack_list.append(compute_jaccard(binary_niblack, gt))

        # SAUVOLA 
        thresh_s = threshold_sauvola(green_blur, window_size, k=k_sauvola)

        binary_sauvola = green_blur < thresh_s
        binary_sauvola = np.logical_and(binary_sauvola, mask)

        sens_sauvola_list.append(compute_sensitivity(binary_sauvola, gt))
        dice_sauvola_list.append(compute_dice(binary_sauvola, gt))
        jacc_sauvola_list.append(compute_jaccard(binary_sauvola, gt))

        # Save one sample result
        if not saved_sample:
            cv2.imwrite("result_original.png", image)
            cv2.imwrite("result_gt.png", gt.astype(np.uint8) * 255)
            cv2.imwrite("result_niblack.png", binary_niblack.astype(np.uint8) * 255)
            cv2.imwrite("result_sauvola.png", binary_sauvola.astype(np.uint8) * 255)
            saved_sample = True


# FINAL RESULTS

print("")
print("NIBLACK RESULTS")
print("Average Sensitivity:", round(np.mean(sens_niblack_list), 4))
print("Average Dice:", round(np.mean(dice_niblack_list), 4))
print("Average Jaccard:", round(np.mean(jacc_niblack_list), 4))
print("")
print("SAUVOLA RESULTS")
print("Average Sensitivity:", round(np.mean(sens_sauvola_list), 4))
print("Average Dice:", round(np.mean(dice_sauvola_list), 4))
print("Average Jaccard:", round(np.mean(jacc_sauvola_list), 4))
