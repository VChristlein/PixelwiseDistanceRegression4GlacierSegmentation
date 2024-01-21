import math
import numpy as np
from pathlib import Path
from skimage import io
import cv2
from tqdm import tqdm
from skimage import color
import os
import argparse

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--pred_path', type=str, help='Path to the predicted test images')
parser.add_argument('--test_path', type=str, help='Path to test images')
parser.add_argument('--save_path', type=str, help='Path to save the evaluation')
parser.add_argument('--method', type=str, help='Set the method that was used for prediction')

# ['sec_model/dist_dice',
# 'sec_model/dist_bce',
# 'sec_model/dice',
# 'front',
# 'crf',
# 'enze',
# 'sec_model/bce',
# 'front',
# 'front_dilate',
# 'patch_front_dilate',
# 'patch_front',
# 'patch_zone'
# ]

args = parser.parse_args()

test_path = args.test_path
pred_path = args.pred_path
save_path = args.save_path
method = args.method

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load images
images = list()
refs = list()
preds = list()
for fname in tqdm(sorted(Path(test_path).rglob('*.png'))):
    if "front.png" in fname.name:
        refs.append(fname)
    else:
        images.append(fname)

s_term = 'bin.png' if 'sec_model' in pred_path else 'pred.png'
for filename in sorted(Path(pred_path).rglob('*.png')):

    if s_term in filename.name:
        preds.append(filename)


def relaxed_dice(y_true, y_pred, ksize=5) -> np.float64:
    """
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    if ksize == 0:
        y_pred_dilate = y_pred
        y_true_dilate = y_true
    else:
        y_pred_dilate = cv2.dilate(y_pred, kernel, iterations=1)
        y_true_dilate = cv2.dilate(y_true, kernel, iterations=1)

    overlap = np.sum(y_pred_dilate * y_true_dilate)
    union = np.sum(y_pred_dilate) + np.sum(y_true_dilate)

    score = 2 * overlap / union
    return score


def relaxed_IoU(y_true, y_pred, ksize=5) -> np.float64:
    """
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    if ksize == 0:
        y_pred_dilate = y_pred
        y_true_dilate = y_true
    else:
        y_pred_dilate = cv2.dilate(y_pred, kernel, iterations=1)
        y_true_dilate = cv2.dilate(y_true, kernel, iterations=1)

    overlap = np.bitwise_and(y_pred_dilate, y_true_dilate).sum()
    union = np.bitwise_or(y_pred_dilate, y_true_dilate).sum()

    score = overlap / union
    return score


def qualitative_eval(method, relax):
    dice_list = list()
    iou_list = list()

    for i, _ in enumerate(refs):

        pred = io.imread(preds[i])

        ref = io.imread(refs[i])
        ref = cv2.resize(ref, shape)

        image = io.imread(images[i])
        image = color.gray2rgb(image)
        image = cv2.resize(image, shape)

        resolution = int(refs[i].stem.split('_')[2])
        iterations = int(50 / resolution)
        shape = tuple(reversed(pred.shape))
        k = int((max(shape) / 512) * 2)

        # morphological change width of prediction
        if k > 0:
            if "threshold" in method:
                pred = cv2.dilate(pred, np.ones((k, k), np.uint8), iterations=2)
            elif "crf" in method:
                pred = cv2.erode(pred, np.ones((4, 4), np.uint8), iterations=1)
                pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))
            elif "second_model" in method:
                pred = cv2.erode(pred, np.ones((2, 2), np.uint8), iterations=iterations)

        # calculate kernel size for each image resolution individual
        ksize = math.ceil(relax / resolution)

        iou = relaxed_IoU(ref, pred, ksize=ksize)
        dice = relaxed_dice(ref / 255, pred / 255, ksize=ksize)

        iou_list.append(iou)
        dice_list.append(dice)

        # dilate reference for better visual
        if k > 1:
            # pred = cv2.dilate(pred, np.ones((k,k), np.uint8))
            ref = cv2.dilate(ref, np.ones((k, k), np.uint8), iterations=1)

        image[ref > 0] = [0, 255, 0]
        image[pred > 0] = [255, 0, 0]
        cor = ref * pred
        image[cor > 0] = [255, 255, 0]

        # only save qualitative image once, e.g. if tolerance=0
        if tolerance == 0:
            io.imsave(f'{save_path}/{refs[i].stem[:-6]}.png', image)

    print('Dice mean: {:.5f}, std: {:.5f}, median: {:.5f}'.format(np.mean(dice_list), np.std(dice_list),
                                                                  np.median(dice_list)))
    print('IOU mean: {:.5f}, std: {:.5f}, median: {:.5f}'.format(np.mean(iou_list), np.std(iou_list),
                                                                 np.median(iou_list)))
    print()
    np.save(f'{save_path}/dice_{relax}.npy', dice_list)
    np.save(f'{save_path}/iou_{relax}.npy', iou_list)


tolerance = [0, 50, 100, 150, 200, 250]

for t in tolerance:
    print(f"Tolerance: {t}")
    qualitative_eval(method=method, relax=t)
