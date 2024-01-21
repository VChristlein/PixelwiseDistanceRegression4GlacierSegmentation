from tqdm import tqdm
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import os
import imageio
from util.distance_map import *
from util.canny import auto_canny
import numpy as np
from util.crf import *
import cv2
from pathlib import Path
from skimage import io
import sys
from scipy.spatial import distance
from sklearn.metrics import r2_score, mean_squared_error
from model.losses import *
import tensorflow.keras.backend as K

TARGET_SIZE = 512
GAMMA = 7

TEST_PATH = f'data_{TARGET_SIZE}/test'
MODEL_PATH = 'saved_model/test_model'

OUT_PATH = 'report'

dependencies = {
    'r2_score': None,
    'cross_dice': cross_dice,
    'distance_weighted_dice_loss': distance_weighted_dice_loss,
    'distance_weighted_bce_loss': distance_weighted_bce_loss,
    'dice_loss': dice_loss
}


def print_hist(save_fig=False, model_path=MODEL_PATH, out_path=OUT_PATH):
    with open(f'{model_path}/history.pickle', 'rb') as f:
        history = pickle.load(f)

    history_path = f'{out_path}/history'

    # create paths
    if not os.path.exists(history_path):
        os.makedirs(history_path)

    print(history.keys())
    print(f'Epochs: {len(history["loss"])}')

    print(f'Loss min: train {min(history["loss"])}, val {min(history["val_loss"])}')

    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.plot(history['loss'], 'X-', label='training loss', linewidth=4.0)
    plt.plot(history['val_loss'], 'o-', label='validation loss', linewidth=4.0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.ylim((0, 1))
    plt.grid(which='minor', linestyle='--')
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{history_path}/loss.png', bbox_inches='tight', format='png', dpi=200)
    plt.show()

    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.plot(history['loss'], 'X-', label='training loss', linewidth=4.0)
    plt.plot(history['val_loss'], 'o-', label='validation loss', linewidth=4.0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.yscale('log')
    plt.ylim((10 ** -10, 1))
    plt.grid(which='minor', linestyle='--')
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{history_path}/loss_log.png', bbox_inches='tight', format='png', dpi=200)
    plt.show()


'''
def front_distance(pred, ref):
    coords_pred = np.where(pred > 0)
    coords_ref = np.where(ref > 0)

    def scipy_distance(data):
        a, b = data[0]
        return list(map(distance.euclidean, a, b))

    scipy_distance()


def mean_front_distance(pred, ref):
    coords_pred = np.where(pred > 0)
    coords_ref = np.where(ref > 0)

    mean_pred = np.mean(coords_pred, axis=1)
    mean_ref = np.mean(coords_ref, axis=1)

    dist = distance.euclidean(mean_pred, mean_ref)

    return dist
'''


def evaluate_gamma():
    mse_list, r2_list, distance_list = list(), list(), list()
    gamma = range(1, 11)

    for g in gamma:
        r2_scores = np.load(f'report/{g}/quantitative/r2_score.npy')
        mse_scores = np.load(f'report/{g}/quantitative/mse_score.npy')
        distance_scores = np.load(f'report/{g}/quantitative/distance_score.npy', allow_pickle=True)

        my_dist_score = list()
        for scores in distance_scores:
            my_dist_score.append(np.mean(scores))

        r2_list.append(r2_scores)
        mse_list.append(mse_scores)
        distance_list.append(my_dist_score)

    plt.boxplot(r2_list)
    plt.ylim((0, 1))
    plt.xlabel("Gamma")
    plt.ylabel("R2-score")
    plt.savefig(f'{OUT_PATH}/r2_score_gamma.png', bbox_inches='tight', format='png', dpi=200)
    plt.show()

    plt.boxplot(mse_list)
    plt.ylim((0, 0.01))
    plt.xlabel("Gamma")
    plt.ylabel("MSE-score")
    plt.savefig(f'{OUT_PATH}/mse_score_gamma.png', bbox_inches='tight', format='png', dpi=200)
    plt.show()

    plt.boxplot(distance_list)
    plt.ylim((0, 0.1))
    plt.xlabel("Gamma")
    plt.ylabel("Distance-score")
    plt.savefig(f'{OUT_PATH}/distance_score_gamma.png', bbox_inches='tight', format='png', dpi=200)
    plt.show()


def calculate_distance(ref, pred):
    # advanced distance algorithm
    img_1 = ref
    img_2 = pred
    coords_1 = np.where(img_1 > 0)
    coords_2 = np.where(img_2 > 0)

    coords_1 = np.array(coords_1).astype(np.float64)
    coords_2 = np.array(coords_2).astype(np.float64)

    coords_1[0] = coords_1[0] / img_1.shape[0]
    coords_1[1] = coords_1[1] / img_1.shape[1]

    coords_2[0] = coords_2[0] / img_1.shape[0]
    coords_2[1] = coords_2[1] / img_1.shape[1]

    coords_1 = coords_1.transpose()
    coords_2 = coords_2.transpose()

    mins = list()
    for c1 in coords_1:
        dist = list()
        for c2 in coords_2:
            dist.append(distance.euclidean(c1, c2))

        mins.append(min(dist))

    return mins


def evaluate(gamma=GAMMA, model_path=MODEL_PATH, second_model_path=None, test_path=TEST_PATH, out_path=None, crf=False, enze_model=False):
    print("Evaluating model...")

    out_path = f'report/{gamma}' if gamma else 'report'
    if second_model_path:
        out_path = f'report/sec/{gamma}' if gamma else 'report/sec'

    qualitative_path = f'{out_path}/qualitative'
    quantitative_path = f'{out_path}/quantitative'

    # create paths
    if not os.path.exists(f'{qualitative_path}/front'):
        os.makedirs(f'{qualitative_path}/front')
    if not os.path.exists(f'{qualitative_path}/distance_map'):
        os.makedirs(f'{qualitative_path}/distance_map')
    if not os.path.exists(quantitative_path):
        os.makedirs(quantitative_path)

    # create hist
    print_hist(save_fig=True, model_path=model_path, out_path=quantitative_path)

    # load model
    distance_map_model = load_model(f'{model_path}/unet.hdf5', custom_objects=dependencies)

    # calculate best threshold
    def calculate_threshold(val_path=None, front_model=None):
        batch_size = 3

        images = list()
        refs = list()
        for filename in Path(val_path, 'images').rglob('*.png'):
            img = io.imread(filename, as_gray=True)
            img = img / 255

            reference = f'{val_path}/masks_front/{filename.name}'
            reference = io.imread(reference, as_gray=True)
            reference = reference / 255

            images.append(img)
            refs.append(reference)

        images = np.array(images)
        images = np.reshape(images, images.shape + (1,))
        refs = np.array(refs)

        distance_map_pred = distance_map_model.predict(images, batch_size=batch_size, verbose=1)
        preds = front_model.predict(distance_map_pred, batch_size=batch_size, verbose=1)

        pred_y = np.reshape(preds, preds.shape[:-1])
        ref_y = refs

        def overall_dice(preds, refs):
            n = preds.shape[0]

            preds = preds.reshape(n, -1)
            refs = refs.reshape(n, -1)

            intersect = np.sum(preds * refs, axis=1)
            union = np.sum(preds + refs, axis=1)

            return 2 * intersect / union

        dices = list()

        thrs = np.arange(0.01, 1, 0.01)
        for i in tqdm(thrs):
            new_pred = np.where(pred_y < i, 0, 1)
            dice = overall_dice(new_pred, ref_y)

            dice_mean = np.mean(dice)
            print(dice_mean)

            dices.append(dice_mean)

        dices = np.array(dices)
        best_dice = dices.max()
        best_thr = thrs[dices.argmax()]

        plt.figure()
        plt.plot(thrs, dices)
        plt.vlines(x=best_thr, ymin=dices.min(), ymax=dices.max())
        plt.text(best_thr + 0.03, best_dice - 0.01, f'DICE = {best_dice:.3f}', fontsize=14)
        plt.title('Dice Threshold')
        plt.savefig(f'{quantitative_path}/threshold.png')
        plt.show()

        return best_thr

    # for second model evaluation
    if second_model_path:
        front_model = load_model(f'{second_model_path}/unet.hdf5', custom_objects=dependencies)
        val_path = f'{test_path}/../val'
        threshold = calculate_threshold(val_path, front_model)
        print(f'Threshold: {threshold}')

    # for enze model evaluation
    if enze_model:
        val_path = f'{test_path}/../val'
        threshold = calculate_threshold(val_path, distance_map_model)
        print(f'Threshold: {threshold}')

    r2_scores, mse_scores, la_scores, relaxed_dice_scores = list(), list(), list(), list()
    distances = list()

    def relaxed_dice(y_true, y_pred, size=10):
        kernel = np.ones((size, size), np.uint8)
        y_true_dilate = cv2.dilate(y_true, kernel, iterations=1)

        intersection = 2 * np.sum(y_pred * y_true_dilate)
        union = np.sum(y_pred) + np.sum(y_true)
        return intersection / union

    for filename in tqdm(sorted(Path(test_path, 'images').rglob('*.png'))):
        # load image
        image = io.imread(filename)

        original_shape = tuple(reversed(image.shape))

        # resize to model target size and normalize
        image = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE))  # interpolation=cv2.INTER_AREA)
        image = image / 255

        # predict distance map
        distance_map_pred = distance_map_model.predict((image[np.newaxis, :, :, np.newaxis]))
        distance_map_pred = distance_map_pred[0, :, :, 0]

        distance_map_pred = cv2.normalize(src=distance_map_pred, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        distance_map_pred_norm = cv2.normalize(src=distance_map_pred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # resize to original size and normalize
        distance_map_pred_res = cv2.resize(distance_map_pred, original_shape, interpolation=cv2.INTER_CUBIC)
        distance_map_pred_res_norm = cv2.normalize(src=distance_map_pred_res, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # save prediction
        # cv2.imwrite(f'{qualitative_path}/distance_map/{filename.stem}_pred.png', distance_map_pred_norm)
        if enze_model:
            cv2.imwrite(f'{qualitative_path}/front/{filename.stem}_pred_res.png', distance_map_pred_res_norm)

            distance_map_pred_res[distance_map_pred_res < threshold] = 0
            distance_map_pred_res[distance_map_pred_res >= threshold] = 1

            distance_map_pred_res_norm_bin = cv2.normalize(src=distance_map_pred_res, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(f'{qualitative_path}/front/{filename.stem}_pred_res_bin.png', distance_map_pred_res_norm_bin)
        else:
            cv2.imwrite(f'{qualitative_path}/distance_map/{filename.stem}_pred_res.png', distance_map_pred_res_norm)

        # load reference
        reference = f'{test_path}/masks_front/{filename.stem}_front.png'
        reference = io.imread(reference)

        # create reference distance map
        distance_map_ref_res_norm = create_distance_map(reference, gamma=gamma)

        distance_map_ref = cv2.resize(distance_map_ref_res_norm, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_CUBIC)

        distance_map_ref_norm = cv2.normalize(src=distance_map_ref, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # save reference
        # cv2.imwrite(f'{qualitative_path}/distance_map/{filename.stem}_ref.png', distance_map_ref_norm)
        cv2.imwrite(f'{qualitative_path}/distance_map/{filename.stem}_ref_res.png', distance_map_ref_res_norm)

        # predict mask front
        if second_model_path:
            # second unet
            mask_front_pred = front_model.predict(distance_map_pred[np.newaxis, :, :, np.newaxis])
            mask_front_pred = mask_front_pred[0, :, :, 0]

            # resize to original size
            mask_front_pred_res = cv2.resize(mask_front_pred, original_shape, interpolation=cv2.INTER_CUBIC)

        elif crf:
            # crf
            mask_front_pred_res = crf(np.expand_dims(image, axis=-1), distance_map_pred_res_norm/255)
        else:
            #threshold
            mask_front_pred_res = distance_map_to_front(distance_map_pred_res_norm)

        # normalize
        mask_front_pred_res_norm = cv2.normalize(src=mask_front_pred_res, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if second_model_path:
            mask_front_pred_res[mask_front_pred_res < threshold] = 0
            mask_front_pred_res[mask_front_pred_res >= threshold] = 1

        # binary form
        mask_front_pred_res_norm_bin = cv2.normalize(src=mask_front_pred_res, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mask_front_pred_res_norm_bin_morph = cv2.morphologyEx(mask_front_pred_res_norm_bin, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
        mask_front_pred_res_norm_bin_morph = cv2.morphologyEx(mask_front_pred_res_norm_bin_morph, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))

        cv2.imwrite(f'{qualitative_path}/front/{filename.stem}_ref.png', reference)
        cv2.imwrite(f'{qualitative_path}/front/{filename.stem}_pred.png', mask_front_pred_res_norm)
        cv2.imwrite(f'{qualitative_path}/front/{filename.stem}_pred_bin.png', mask_front_pred_res_norm_bin)
        cv2.imwrite(f'{qualitative_path}/front/{filename.stem}_pred_bin_mor.png', mask_front_pred_res_norm_bin_morph)

        # relaxed dice score
        score = relaxed_dice(reference, mask_front_pred_res_norm_bin)
        relaxed_dice_scores.append(score)

    plt.figure()
    plt.hist(relaxed_dice_scores)
    plt.savefig(f'{quantitative_path}/rel_dice_hist.png', bbox_inches='tight', format='png', dpi=200)
    plt.show()

    with open(f'{quantitative_path}/relaxed_dice_score.npy', 'wb') as f:
        np.save(f, la_scores)

    '''
        # pixel evaluation
        kernel = np.ones((10, 10), np.uint8)
        reference_dilate = cv2.dilate(reference, kernel, iterations=1)
        pixel_ratio = cv2.bitwise_and(reference_dilate, mask_front_pred_res)

        line_accuracy_score = np.count_nonzero(pixel_ratio) / np.count_nonzero(mask_front_pred_res_norm_bin)
        la_scores.append(line_accuracy_score)

        # calculate scores
        r2_scores.append(r2_score(distance_map_ref, distance_map_pred))
        mse_scores.append(mean_squared_error(distance_map_ref, distance_map_pred))
        distances.append(calculate_distance(reference, mask_front_pred))

    with open(f'{quantitative_path}/line_accuracy_score.npy', 'wb') as f:
        np.save(f, la_scores)

    with open(f'{quantitative_path}/r2_score.npy', 'wb') as f:
        np.save(f, r2_scores)

    with open(f'{quantitative_path}/mse_score.npy', 'wb') as f:
        np.save(f, mse_scores)

    with open(f'{quantitative_path}/distance_score.npy', 'wb') as f:
        np.save(f, distances)
    '''

    K.clear_session()

# print_hist()
# evaluate(gamma=7, model_path="saved_model/20200630-120208_7", second_model_path="saved_model/20200718-134130_3_second", test_path="data_512_7/test")
# evaluate_gamma()
