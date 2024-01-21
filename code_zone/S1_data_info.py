import pandas as pd
import Amir_utils
import imageio
import cv2
from pathlib import Path
from scipy import ndimage
import os
import numpy as np
from tqdm import tqdm
import time
import sys
import matplotlib.pyplot as plt

df_train = pd.read_csv('data/train_images.csv')
df_val = pd.read_csv('data/validation_images.csv')
df_test = pd.read_csv('data/test_images.csv')

PATCH_SIZE = 256
STRIDE = (PATCH_SIZE, PATCH_SIZE)

patch_counter_train = 0
patch_counter_val = 0
patch_counter_test = 0
AUGMENTATION = True
USE_ZERO_PADDING = True

train_rotations = [0, 90, 180, 270]


def apply_augmentation(image, zone, line):
    """
    applies data augmentation

    :param image: original image
    :param zone: image of masks zone
    :param line: image of masked front
    :return: lists of augmented images, zones and fronts
    """

    images, zones, lines = list(), list(), list()

    for rotation in train_rotations:
        rot_image = ndimage.rotate(image, rotation)
        rot_zone = ndimage.rotate(zone, rotation)
        rot_line = ndimage.rotate(line, rotation)

        images.append(rot_image)
        zones.append(rot_zone)
        lines.append(rot_line)

        images.append(np.flip(rot_image))
        zones.append(np.flip(rot_zone))
        lines.append(np.flip(rot_line))

    return images, zones, lines


START = time.time()

print('Generating Training Patches...')

# train path
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/train/images'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/train/images')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_zones'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_zones')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_lines'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_lines')))

for _, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):
    image_data = imageio.imread(row['images'])
    masks_line = imageio.imread(row['lines'])
    masks_zone_tmp = imageio.imread(row['masks'])
    masks_zone_tmp[masks_zone_tmp == 127] = 0
    masks_zone_tmp[masks_zone_tmp == 254] = 255

    # Here, before patch extraction, do the preprocessing, e.g., bilateral filter and/or contrast enhancement

    if USE_ZERO_PADDING:
        image_data = cv2.copyMakeBorder(image_data, 0, (PATCH_SIZE - image_data.shape[0]) % PATCH_SIZE, 0,
                                        (PATCH_SIZE - image_data.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
        masks_line = cv2.copyMakeBorder(masks_line, 0, (PATCH_SIZE - masks_line.shape[0]) % PATCH_SIZE, 0,
                                        (PATCH_SIZE - masks_line.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
        masks_zone_tmp = cv2.copyMakeBorder(masks_zone_tmp, 0, (PATCH_SIZE - masks_zone_tmp.shape[0]) % PATCH_SIZE, 0,
                                            (PATCH_SIZE - masks_zone_tmp.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
        stride = STRIDE
    else:
        stride = (image_data.shape[0] // PATCH_SIZE, image_data.shape[1] // PATCH_SIZE)
        stride = ((image_data.shape[0] - PATCH_SIZE) // stride[0], (image_data.shape[1] - PATCH_SIZE) // stride[1])

    p_masks_zone, i_masks_zone = Amir_utils.extract_grayscale_patches(masks_zone_tmp, (PATCH_SIZE, PATCH_SIZE), stride=stride)
    p_img, i_img = Amir_utils.extract_grayscale_patches(image_data, (PATCH_SIZE, PATCH_SIZE), stride=stride)
    p_masks_line, i_masks_line = Amir_utils.extract_grayscale_patches(masks_line, (PATCH_SIZE, PATCH_SIZE), stride=stride)

    for j in range(p_masks_zone.shape[0]):
        # if np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) > 0.05 and np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) < 0.95: # only those patches that has both background and foreground
        # if np.count_nonzero(p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) >= 0 and np.count_nonzero(p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) <= 1:

        if AUGMENTATION:

            # apply augmentation
            images, masks_zone, masks_line = apply_augmentation(p_img[j], p_masks_zone[j], p_masks_line[j])

            for idx, _ in enumerate(images):
                cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/train/images/' + str(patch_counter_train) + '.png')), images[idx])
                cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_zones/' + str(patch_counter_train) + '.png')), masks_zone[idx])
                cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_lines/' + str(patch_counter_train) + '.png')), masks_line[idx])
                patch_counter_train += 1

        else:
            cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/train/images/' + str(patch_counter_train) + '.png')), p_img[j])
            cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_zones/' + str(patch_counter_train) + '.png')), p_masks_zone[j])
            cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/train/masks_lines/' + str(patch_counter_train) + '.png')), p_masks_line[j])
            patch_counter_train += 1
    # store the name of the file that the patch is from in a list as well

print('Generating Validation Patches...')

# validation path
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/val/images'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/val/images')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_zones'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_zones')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_lines'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_lines')))

for _, row in tqdm(df_val.iterrows(), total=df_val.shape[0]):

    image_data = imageio.imread(row['images'])
    masks_line = imageio.imread(row['lines'])
    masks_zone_tmp = imageio.imread(row['masks'])
    masks_zone_tmp[masks_zone_tmp == 127] = 0
    masks_zone_tmp[masks_zone_tmp == 254] = 255

    # Here, before patch extraction, do the preprocessing, e.g., bilateral filter and/or contrast enhancement

    if USE_ZERO_PADDING:
        image_data = cv2.copyMakeBorder(image_data, 0, (PATCH_SIZE - image_data.shape[0]) % PATCH_SIZE, 0,
                                        (PATCH_SIZE - image_data.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
        masks_line = cv2.copyMakeBorder(masks_line, 0, (PATCH_SIZE - masks_line.shape[0]) % PATCH_SIZE, 0,
                                        (PATCH_SIZE - masks_line.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
        masks_zone_tmp = cv2.copyMakeBorder(masks_zone_tmp, 0, (PATCH_SIZE - masks_zone_tmp.shape[0]) % PATCH_SIZE, 0,
                                            (PATCH_SIZE - masks_zone_tmp.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
        stride = STRIDE
    else:
        stride = (image_data.shape[0] // PATCH_SIZE, image_data.shape[1] // PATCH_SIZE)
        stride = ((image_data.shape[0] - PATCH_SIZE) // stride[0], (image_data.shape[1] - PATCH_SIZE) // stride[1])

    p_masks_zone, i_masks_zone = Amir_utils.extract_grayscale_patches(masks_zone_tmp, (PATCH_SIZE, PATCH_SIZE), stride=stride)
    p_img, i_img = Amir_utils.extract_grayscale_patches(image_data, (PATCH_SIZE, PATCH_SIZE), stride=stride)
    p_masks_line, i_masks_line = Amir_utils.extract_grayscale_patches(masks_line, (PATCH_SIZE, PATCH_SIZE), stride=stride)
    for j in range(p_masks_zone.shape[0]):
        # if np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) > 0.05 and np.count_nonzero(p_masks_zone[j])/(PATCH_SIZE*PATCH_SIZE) < 0.95: # only those patches that has both background and foreground
        # if np.count_nonzero(p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) >= 0 and np.count_nonzero(p_masks_zone[j]) / (PATCH_SIZE * PATCH_SIZE) <= 1:

        cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/val/images/' + str(patch_counter_val) + '.png')), p_img[j])
        cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_zones/' + str(patch_counter_val) + '.png')), p_masks_zone[j])
        cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/val/masks_lines/' + str(patch_counter_val) + '.png')), p_masks_line[j])
        patch_counter_val += 1

print('Generating Test Patches...')

# test path
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/test/images'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/test/images')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/test/masks_zones'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/test/masks_zones')))
if not os.path.exists(str(Path('data_' + str(PATCH_SIZE) + '/test/masks_lines'))):
    os.makedirs(str(Path('data_' + str(PATCH_SIZE) + '/test/masks_lines')))

for _, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
    # extract image data
    image_data = imageio.imread(row['images'])
    masks_line = imageio.imread(row['lines'])
    masks_zone_tmp = imageio.imread(row['masks'])
    masks_zone_tmp[masks_zone_tmp == 127] = 0
    masks_zone_tmp[masks_zone_tmp == 254] = 255

    # recreate filename
    filename = row['images'][38:-4].split('/')[2]

    # write data to file
    cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/test/images/' + filename + '.png')), image_data)
    cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/test/masks_zones/' + filename + '_zones.png')), masks_zone_tmp)
    cv2.imwrite(str(Path('data_' + str(PATCH_SIZE) + '/test/masks_lines/' + filename + '_front.png')), masks_line)
    patch_counter_test += 1

END = time.time()

print(f'Training data: \t\t{patch_counter_train}')
print(f'Validation data: \t{patch_counter_val}')
print(f'Test images: \t\t{patch_counter_test}')
print(f'Process finished: {END - START} seconds')
