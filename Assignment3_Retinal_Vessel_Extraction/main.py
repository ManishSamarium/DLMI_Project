import os
import cv2
import numpy as np
from skimage.filters import threshold_niblack, threshold_sauvola

image_folder = "training/images/"
gt_folder = "training/1st_manual/"
mask_folder = "training/mask/"

window_size = 25
k_niblack = 0.2
k_sauvola = 0.2

sens_niblack_list = []
sens_sauvola_list = []

saved_sample = False

def compute_sensitivity(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    if (tp + fn) == 0:
        return 0
    return tp / (tp + fn)

for file in os.listdir(image_folder):
    if file.endswith(".tif"):

        img_path = os.path.join(image_folder, file)
        gt_path = os.path.join(gt_folder, file.replace("_training.tif", "_manual1.gif"))
        mask_path = os.path.join(mask_folder, file.replace(".tif", "_mask.gif"))

        image = cv2.imread(img_path)
        gt = cv2.imread(gt_path, 0)
        mask = cv2.imread(mask_path, 0)

        if image is None:
            continue

        green = image[:, :, 1]
        green = cv2.GaussianBlur(green, (5, 5), 0)

        gt = gt > 0
        mask = mask > 0

        # ----- Niblack -----
        thresh_n = threshold_niblack(green, window_size, k=k_niblack)

        bin_n1 = green > thresh_n
        bin_n2 = green < thresh_n

        sens_n1 = compute_sensitivity(np.logical_and(bin_n1, mask), gt)
        sens_n2 = compute_sensitivity(np.logical_and(bin_n2, mask), gt)

        if sens_n1 > sens_n2:
            binary_niblack = bin_n1
            sens_niblack_list.append(sens_n1)
        else:
            binary_niblack = bin_n2
            sens_niblack_list.append(sens_n2)

        # ----- Sauvola -----
        thresh_s = threshold_sauvola(green, window_size, k=k_sauvola)

        bin_s1 = green > thresh_s
        bin_s2 = green < thresh_s

        sens_s1 = compute_sensitivity(np.logical_and(bin_s1, mask), gt)
        sens_s2 = compute_sensitivity(np.logical_and(bin_s2, mask), gt)

        if sens_s1 > sens_s2:
            binary_sauvola = bin_s1
            sens_sauvola_list.append(sens_s1)
        else:
            binary_sauvola = bin_s2
            sens_sauvola_list.append(sens_s2)

        if not saved_sample:
            cv2.imwrite("result_niblack.png", binary_niblack.astype(np.uint8)*255)
            cv2.imwrite("result_sauvola.png", binary_sauvola.astype(np.uint8)*255)
            cv2.imwrite("result_gt.png", gt.astype(np.uint8)*255)
            saved_sample = True

print("===================================")
print("Average Sensitivity (Niblack):", round(np.mean(sens_niblack_list), 4))
print("Average Sensitivity (Sauvola):", round(np.mean(sens_sauvola_list), 4))
print("===================================")
