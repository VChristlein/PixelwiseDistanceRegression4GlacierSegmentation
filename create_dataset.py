from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import imageio
import time
from scipy import ndimage
from tqdm import tqdm
import sys
from util.distance_map import create_distance_map
import os
import subprocess
import matplotlib.pyplot as plt
from skimage import io
from tensorflow.keras.models import load_model

TARGET_SIZE = 512
train_rotations = [0, 90, 180, 270]
GAMMA = 7
MORPH_FACTOR = 6


def delete_data(data_dir):
    subprocess.run(['rm', '-rf', data_dir], check=True)


def apply_augmentation(image, front, dist_map):
    """
    applies data augmentation

    :param image: original image
    :param front: image of masked front
    :param dist_map: distance map of the front mask
    :return: lists of augmented images, fronts, distance maps
    """

    aug_images, aug_fronts, aug_dist_maps = list(), list(), list()

    for rotation in train_rotations:
        rot_image = ndimage.rotate(image, rotation)
        rot_fronts = ndimage.rotate(front, rotation)
        rot_dist_map = ndimage.rotate(dist_map, rotation)

        aug_images.append(rot_image)
        aug_fronts.append(rot_fronts)
        aug_dist_maps.append(rot_dist_map)

        aug_images.append(np.flip(rot_image))
        aug_fronts.append(np.flip(rot_fronts))
        aug_dist_maps.append(np.flip(rot_dist_map))

    return aug_images, aug_fronts, aug_dist_maps


def create_dataset(gamma=GAMMA, model_path=None, second_model=False):
    START = time.time()
    print('Creating Dataset...')

    DATA_DIR = f'data_{TARGET_SIZE}_{gamma}'

    #############
    # LOAD DATA #
    #############
    images, masks_front = [], []

    # load new dataset
    for filename in sorted(Path('../front_detection/Dataset2_16.06/').rglob('*.png')):

        fname = filename.as_posix()

        if "_zones.png" in fname:
            continue
        if "_zones_NA.png" in fname:
            continue

        if "_front.png" in fname:
            masks_front.append(filename)
        else:
            images.append(filename)

    # load dataset
    for filename in sorted(Path('../front_detection/training-data-zone/').rglob('*.png')):

        fname = filename.as_posix()

        if "_zones.png" in fname:
            continue

        if "_front.png" in fname:
            masks_front.append(filename)
        else:
            images.append(filename)

    ########################
    # TRAIN/VAL/TEST SPLIT #
    ########################
    data_idx = np.arange(len(images))
    train_idx, test_idx = train_test_split(data_idx, test_size=30, random_state=1)  # 50 images are chosen as the test images
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=1)  # 20% of training data as validation data

    #############
    # SAVE DATA #
    #############
    train_counter, val_counter, test_counter = 0, 0, 0

    # create folder structure
    folder1 = ['train', 'test', 'val']
    folder2 = ['images', 'masks_front', 'distance_maps']

    if second_model:
        # load model for distance_map prediction
        model = load_model(f'{model_path}/unet.hdf5')

        # kernel for masks front dilation
        kernel = np.ones((5, 5), np.uint8)

        # create additional folders
        folder2 += ['pred_distance_maps', 'masks_front_morph']

    for f1 in folder1:
        for f2 in folder2:
            if 'test' in f1 and ('distance_maps' or 'morph') in f2:
                continue
            if not os.path.exists(f'{DATA_DIR}/{f1}/{f2}'):
                os.makedirs(f'{DATA_DIR}/{f1}/{f2}')

    # create dataset
    for i in tqdm(data_idx):
        # load image and mask front
        image = imageio.imread(images[i])
        mask_front = imageio.imread(masks_front[i])

        if i in test_idx:
            cv2.imwrite(f'{DATA_DIR}/test/images/{images[i].name}', image)
            cv2.imwrite(f'{DATA_DIR}/test/masks_front/{masks_front[i].name}', mask_front)
            test_counter += 1
            continue

        # create distance map
        distance_map = create_distance_map(mask_front, gamma=gamma)

        # resize data
        image = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE))
        mask_front = cv2.resize(mask_front, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        distance_map = cv2.resize(distance_map, (TARGET_SIZE, TARGET_SIZE))

        # binary mask front
        _, mask_front = cv2.threshold(mask_front, 1, 255, cv2.THRESH_BINARY)

        if i in train_idx:

            aug_images, aug_masks_front, aug_distance_maps = apply_augmentation(image, mask_front, distance_map)

            for j, _ in enumerate(aug_images):
                cv2.imwrite(f'{DATA_DIR}/train/images/{train_counter}.png', aug_images[j])
                cv2.imwrite(f'{DATA_DIR}/train/masks_front/{train_counter}.png', aug_masks_front[j])
                cv2.imwrite(f'{DATA_DIR}/train/distance_maps/{train_counter}.png', aug_distance_maps[j])

                if second_model:
                    # predict distance map
                    image = aug_images[j]
                    image = image / 255
                    distance_map_pred = model.predict((image[np.newaxis, :, :, np.newaxis]))
                    distance_map_pred = distance_map_pred[0, :, :, 0]

                    distance_map_pred_norm = cv2.normalize(src=distance_map_pred, dst=None, alpha=0,
                                                           beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                    mask_front_morph = cv2.dilate(aug_masks_front[j], kernel, iterations=1)

                    cv2.imwrite(f'{DATA_DIR}/train/pred_distance_maps/{train_counter}.png', distance_map_pred_norm)
                    cv2.imwrite(f'{DATA_DIR}/train/masks_front_morph/{train_counter}.png', mask_front_morph)

                train_counter += 1

        elif i in val_idx:
            cv2.imwrite(f'{DATA_DIR}/val/images/{val_counter}.png', image)
            cv2.imwrite(f'{DATA_DIR}/val/masks_front/{val_counter}.png', mask_front)
            cv2.imwrite(f'{DATA_DIR}/val/distance_maps/{val_counter}.png', distance_map)

            if second_model:
                # predict distance map
                image = image / 255
                distance_map_pred = model.predict((image[np.newaxis, :, :, np.newaxis]))
                distance_map_pred = distance_map_pred[0, :, :, 0]

                distance_map_pred_norm = cv2.normalize(src=distance_map_pred, dst=None, alpha=0,
                                                       beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                mask_front_morph = cv2.dilate(mask_front, kernel, iterations=1)

                cv2.imwrite(f'{DATA_DIR}/val/pred_distance_maps/{val_counter}.png', distance_map_pred_norm)
                cv2.imwrite(f'{DATA_DIR}/val/masks_front_morph/{val_counter}.png', mask_front_morph)

            val_counter += 1

        else:
            sys.exit(f'Index error {i}')

    open(f'{DATA_DIR}/gamma{gamma}', 'a').close()

    END = time.time()

    print(f'Training data: \t\t{train_counter}')
    print(f'Validation data: \t{val_counter}')
    print(f'Test images: \t\t{test_counter}')
    print(f'Process finished: {int(END - START)} seconds')

# create_dataset(gamma=7, model_path='saved_model/test_model', second_model=True)
