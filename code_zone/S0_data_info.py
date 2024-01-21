# -*- coding: utf-8 -*-

import imageio
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imsave
import os
import os.path
import shutil
import Amir_utils
import time
import cv2
import sys
import argparse

parser = argparse.ArgumentParser(description='Dataset Generator')

parser.add_argument('--patch_size', type=int, help='Set the patch_size')
parser.add_argument('--plot_model', action="store_true", help='create model png')
parser.add_argument('--load_model', default=None, type=str, help='load model from given path')
parser.add_argument('--chained_training', default=0, type=int, help='use chained jobs for model training')
args = parser.parse_args()

# %%
data_count = 0
data_all = []
data_names = []

for filename in sorted(Path('../front_detection/Dataset2_16.06/').rglob('*.png')):

    if "_zones_NA.png" in filename.as_posix():
        continue

    data = imageio.imread(filename)
    data_all.append(data)
    data_names.append(filename.as_posix())
    data_count += 1

for filename in sorted(Path('../front_detection/training-data-zone/').rglob('*.png')):
    data = imageio.imread(filename)
    data_all.append(data)
    data_names.append(filename.as_posix())
    data_count += 1
print(f'Found {data_count} image files in total')

# %% bilateral filtering + CLAHE
#####################
# You probably should apply these filters on all the images before extracting the patches and saving them
#####################

img = imageio.imread(data_names[-3])  # a sample image
img_bilateral = cv2.bilateralFilter(img, 20, 80, 80)  # bilateral filter
img_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25)).apply(img_bilateral)  # CLAHE adaptive contrast enhancement

plt.figure(figsize=(12, 8))
plt.title('jafar')
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('original image')
plt.subplot(1, 3, 2)
plt.imshow(img_bilateral, cmap='gray')
plt.title('bilateral filter')
plt.subplot(1, 3, 3)
plt.imshow(img_CLAHE, cmap='gray')
plt.title('CLAHE')
plt.tight_layout()
plt.savefig('image_bilateral_CLAHE.png', bbox_inches='tight', format='png', dpi=200)
plt.show()

# %% list of images and masks: (images_names, masks_names)
images_names, images_names_id = [], []
masks_line_names, masks_line_names_id = [], []
masks_zone_names, masks_zone_names_id = [], []
for i in range(len(data_names)):
    if "_front.png" in data_names[i]:
        masks_line_names.append(data_names[i])
        masks_line_names_id.append(i)
    elif "_zones.png" in data_names[i]:
        masks_zone_names.append(data_names[i])
        masks_zone_names_id.append(i)
    else:
        images_names.append(data_names[i])
        images_names_id.append(i)

print(len(images_names_id))
print(len(masks_line_names_id))
print(len(masks_zone_names_id))
# %% check the quality factor from file names and discard the ones with low quality factor (from 1 to 5, the higher is worse)
# TODO ...
# median filter 5x5

# %% separate train and test images
from sklearn.model_selection import train_test_split

data_idx = np.arange(len(images_names_id))
train_idx, test_idx = train_test_split(data_idx, test_size=30, random_state=1)  # 10 images are chosen as the test images
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=1)  # 20% of training data as validation data

# %% generate patches
START = time.time()
PATCH_SIZE = parser.patch_size
kernel = np.ones((5, 5), np.uint8)

#####
# train path
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/train/images'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/train/images')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_zones'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_zones')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_lines'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_lines')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_lines_dilate'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_lines_dilate')))

STRIDE_train = (PATCH_SIZE, PATCH_SIZE)
patch_counter_train = 0
for i in train_idx:
    masks_zone_tmp = data_all[masks_zone_names_id[i]]
    masks_zone_tmp[masks_zone_tmp == 127] = 0
    masks_zone_tmp[masks_zone_tmp == 254] = 255

    # Here, before patch extraction, do the preprocessing, e.g., bilateral filter and/or contrast enhancement

    p_masks_zone, i_masks_zone = Amir_utils.extract_grayscale_patches(masks_zone_tmp, (PATCH_SIZE, PATCH_SIZE), stride=STRIDE_train)
    p_img, i_img = Amir_utils.extract_grayscale_patches(data_all[images_names_id[i]], (PATCH_SIZE, PATCH_SIZE), stride=STRIDE_train)
    p_masks_line, i_masks_line = Amir_utils.extract_grayscale_patches(data_all[masks_line_names_id[i]], (PATCH_SIZE, PATCH_SIZE), stride=STRIDE_train)

    for j in range(p_masks_zone.shape[0]):
        # if np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) > 0.05 and np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) < 0.95: # only those patches that has both background and foreground
        if np.count_nonzero(p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) >= 0 and np.count_nonzero(p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) <= 1:
            cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/train/images/' + str(patch_counter_train) + '.png')), p_img[j])
            cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_zones/' + str(patch_counter_train) + '.png')), p_masks_zone[j])
            cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_lines/' + str(patch_counter_train) + '.png')), p_masks_line[j])

            p_masks_line_dilate = cv2.dilate(p_masks_line[j], kernel, iterations=1)
            cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_lines_dilate/' + str(patch_counter_train) + '.png')), p_masks_line_dilate)

            patch_counter_train += 1
            # store the name of the file that the patch is from in a list as well

#####
# validation path
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/val/images'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/val/images')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_zones'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_zones')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_lines'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_lines')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_lines_dilate'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_lines_dilate')))

STRIDE_val = (PATCH_SIZE, PATCH_SIZE)
patch_counter_val = 0
for i in val_idx:
    masks_zone_tmp = data_all[masks_zone_names_id[i]]
    masks_zone_tmp[masks_zone_tmp == 127] = 0
    masks_zone_tmp[masks_zone_tmp == 254] = 255

    # Here, before patch extraction, do the preprocessing, e.g., bilateral filter and/or contrast enhancement

    p_masks_zone, i_masks_zone = Amir_utils.extract_grayscale_patches(masks_zone_tmp, (PATCH_SIZE, PATCH_SIZE), stride=STRIDE_val)
    p_img, i_img = Amir_utils.extract_grayscale_patches(data_all[images_names_id[i]], (PATCH_SIZE, PATCH_SIZE), stride=STRIDE_val)
    p_masks_line, i_masks_line = Amir_utils.extract_grayscale_patches(data_all[masks_line_names_id[i]], (PATCH_SIZE, PATCH_SIZE), stride=STRIDE_val)

    for j in range(p_masks_zone.shape[0]):
        # if np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) > 0.05 and np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) < 0.95: # only those patches that has both background and foreground
        if np.count_nonzero(p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) > 0 and np.count_nonzero(p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) < 1:
            cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/val/images/' + str(patch_counter_val) + '.png')), p_img[j])
            cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_zones/' + str(patch_counter_val) + '.png')), p_masks_zone[j])
            cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_lines/' + str(patch_counter_val) + '.png')), p_masks_line[j])

            p_masks_line_dilate = cv2.dilate(p_masks_line[j], kernel, iterations=1)
            cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_lines_dilate/' + str(patch_counter_val) + '.png')), p_masks_line_dilate)

            patch_counter_val += 1
            # store the name of the file that the patch is from in a list as well

#####
# test path
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/test/images'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/test/images')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/test/masks_zones'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/test/masks_zones')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/test/masks_lines'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/test/masks_lines')))

STRIDE_test = (PATCH_SIZE, PATCH_SIZE)
patch_counter_test = 0
for i in test_idx:
    masks_zone_tmp = data_all[masks_zone_names_id[i]]
    masks_zone_tmp[masks_zone_tmp == 127] = 0
    masks_zone_tmp[masks_zone_tmp == 254] = 255

    cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/test/images/' + Path(images_names[i]).name)), data_all[images_names_id[i]])
    cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/test/masks_zones/' + Path(masks_zone_names[i]).name)), masks_zone_tmp)
    cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/test/masks_lines/' + Path(masks_line_names[i]).name)), data_all[masks_line_names_id[i]])
    patch_counter_test += 1
# STRIDE_test = (PATCH_SIZE,PATCH_SIZE)
# patch_counter_test = 0
# for i in test_idx: 
#     masks_zone_tmp = data_all[masks_zone_names_id[i]]
#     masks_zone_tmp[masks_zone_tmp==127] = 0
#     masks_zone_tmp[masks_zone_tmp==254] = 255

#     p_masks_zone, i_masks_zone = Amir_utils.extract_grayscale_patches(masks_zone_tmp, (PATCH_SIZE,PATCH_SIZE), stride = STRIDE_test)
#     p_img, i_img = Amir_utils.extract_grayscale_patches(data_all[images_names_id[i]], (PATCH_SIZE,PATCH_SIZE), stride = STRIDE_test)
#     p_masks_line, i_masks_line = Amir_utils.extract_grayscale_patches(data_all[masks_line_names_id[i]], (PATCH_SIZE,PATCH_SIZE), stride = STRIDE_test)

#     for j in range(p_masks_zone.shape[0]):
#         if np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) > 0.05 and np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) < 0.95:
#             cv2.imwrite(str(Path('data_'+str(PATCH_SIZE)+'/test/images/'+str(patch_counter_test)+'.png')), p_img[j])
#             cv2.imwrite(str(Path('data_'+str(PATCH_SIZE)+'/test/masks_zones/'+str(patch_counter_test)+'.png')), p_masks_zone[j])
#             cv2.imwrite(str(Path('data_'+str(PATCH_SIZE)+'/test/masks_lines/'+str(patch_counter_test)+'.png')), p_masks_line[j])
#             patch_counter_test += 1
#             # store the name of the file that the patch is from in a list as well

#####

END = time.time()
print(f'Training data: \t{patch_counter_train}')
print(f'Validation data: \t{patch_counter_val}')
print(f'Test images: \t{patch_counter_test}')
print(f'Process finished:{END - START} seconds')
