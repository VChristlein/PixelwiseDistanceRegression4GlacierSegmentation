# %%
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from model.metrics import *
from model.losses import *
import skimage.io as io
import sys
import Amir_utils
import cv2
import os
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from data import *
from sklearn import metrics
import json
from skimage.measure import label
from skimage import feature
from scipy import ndimage
import argparse
import time

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--model_path', type=str, help='path to trained model')
parser.add_argument('--patch_size', default=256, type=int, help='patch size (integer value)')
parser.add_argument('--out_path', default=f'output/{time.time()}', type=str, help='output path for results')
parser.add_argument('--zone_model', default=0, type=int, help='evaluate zone model (0/1)')
args = parser.parse_args()


def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC


'''
print(os.getcwd())
os.chdir('code_zone')

with open('config.yaml', 'r') as f:
    config = yaml.load(f)
    # print(config['model']['name'])
'''

timestamp = None
PATCH_SIZE = args.patch_size
model_path = args.model_path
out_path = args.out_path

test_path = str(Path('data_' + str(PATCH_SIZE) + '/test/'))
val_path = str(Path('data_' + str(PATCH_SIZE) + '/val/'))

# load model
dependencies = {
    'dice_coef': dice_coef,
    'iou_coef': iou_coef,
    'binary_focal_loss_fixed': focal_loss(),
    'dice_loss': dice_loss
}

model = load_model(f'{model_path}/unet0.hdf5', custom_objects=dependencies)


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


make_path(out_path)


def relaxed_dice(y_true, y_pred, ksize=5) -> np.float64:
    """
    """
    kernel = np.ones((ksize, ksize), np.uint8)
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
    y_pred_dilate = cv2.dilate(y_pred, kernel, iterations=1)
    y_true_dilate = cv2.dilate(y_true, kernel, iterations=1)

    overlap = np.bitwise_and(y_pred_dilate, y_true_dilate).sum()
    union = np.bitwise_or(y_pred_dilate, y_true_dilate).sum()

    score = overlap / union
    return score


# %%
def print_hist(save_fig=False):
    with open(f'{model_path}/history.pickle', 'rb') as f:
        history = pickle.load(f)

    print('Epoch: ' + str(len(history['loss'])))
    # print('Accuracy: train {:.3f}, val {:.3f}'.format(max(history['acc']), max(history['val_acc'])))
    print('Loss: train {:.3f}, val {:.3f}'.format(min(history['loss']), min(history['val_loss'])))
    # print('Dice: train {:.3f}, val {:.3f}'.format(max(history['dice_coef']), max(history['val_dice_coef'])))

    save_path = f'{out_path}/history'
    make_path(save_path)

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
        plt.savefig(f'{save_path}/loss.png', bbox_inches='tight', format='png', dpi=200)
    plt.show()

    '''
    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.plot(history['acc'], 'X-', label='train acc', linewidth=4.0)
    plt.plot(history['val_acc'], 'o-', label='val acc', linewidth=4.0)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.ylim((0, 1))
    plt.grid(which='minor', linestyle='--')
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{save_path}/accuracy.png')
    plt.show()

    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.plot(history['iou_coef'], 'X-', label='train iou coef', linewidth=4.0)
    plt.plot(history['val_iou_coef'], 'o-', label='val iou coef', linewidth=4.0)
    plt.xlabel('epoch')
    plt.ylabel('iou coef')
    plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.ylim((0, 1))
    plt.grid(which='minor', linestyle='--')
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{save_path}/iou.png')
    plt.show()

    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.plot(history['dice_coef'], 'X-', label='train dice', linewidth=4.0)
    plt.plot(history['val_dice_coef'], 'o-', label='val dice', linewidth=4.0)
    plt.xlabel('epoch')
    plt.ylabel('dice coefficient')
    plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.ylim((0, 1))
    plt.grid(which='minor', linestyle='--')
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{save_path}/dice.png')
    plt.show()
    '''


print_hist(save_fig=True)


# sys.exit()


# %%
def calculate_threshold(save_fig=False):
    batch_size = 2  #

    images = list()
    refs = list()
    for filename in Path(val_path, 'images').rglob('*.png'):
        img = io.imread(filename, as_gray=True)
        img = img / 255

        reference = f'{val_path}/masks_zones/{filename.name}'
        reference = io.imread(reference, as_gray=True)
        reference = reference / 255

        images.append(img)
        refs.append(reference)

    images = np.array(images)
    images = np.reshape(images, images.shape + (1,))
    refs = np.array(refs)

    preds = model.predict(images, batch_size=batch_size, verbose=1)

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

    plt.plot(thrs, dices)
    plt.vlines(x=best_thr, ymin=dices.min(), ymax=dices.max())
    plt.text(best_thr + 0.03, best_dice - 0.01, f'DICE = {best_dice:.3f}', fontsize=14)
    plt.title('Dice Threshold')
    if save_fig:
        plt.savefig(f'{out_path}/threshold.png')
    plt.show()

    return best_thr


# %%
def qualitative_evaluation(bar_plot=True, save_fig=False, number=15):
    save_path = f'{out_path}/qualitative/images'
    make_path(save_path)

    threshold = calculate_threshold(save_fig=True)
    print(threshold)

    acc = list()
    sensitivity = list()
    specificity = list()
    ref_list = list()
    pred_list = list()

    i = 0
    for filename in Path(test_path, 'images').rglob('*.png'):
        # load image
        img = io.imread(filename, as_gray=True)
        img = img / 255

        # load reference
        reference = f'{test_path}/masks_zones/{filename.stem}_zones.png'
        reference = io.imread(reference, as_gray=True)

        # pad image
        img_pad = cv2.copyMakeBorder(img, 0, (PATCH_SIZE - img.shape[0]) % PATCH_SIZE, 0,
                                     (PATCH_SIZE - img.shape[1]) % PATCH_SIZE,
                                     cv2.BORDER_CONSTANT)

        p_img, i_img = Amir_utils.extract_grayscale_patches(img_pad, (PATCH_SIZE, PATCH_SIZE),
                                                            stride=(PATCH_SIZE, PATCH_SIZE))
        p_img = np.reshape(p_img, p_img.shape + (1,))

        # predict
        p_img_predicted = model.predict(p_img, batch_size=3)
        p_img_predicted = np.reshape(p_img_predicted, p_img_predicted.shape[:-1])

        # unpad and normalize
        p_img_predicted = Amir_utils.reconstruct_from_grayscale_patches(p_img_predicted, i_img)[0]
        p_img_predicted = p_img_predicted[0:img.shape[0], 0:img.shape[1]]

        # apply threshold from val data
        p_img_predicted[p_img_predicted < threshold] = 0
        p_img_predicted[p_img_predicted >= threshold] = 1


        if args.zone_model:
            plt.imshow(p_img_predicted)
            plt.show()
            scalar = .025
            # print(int(p_img_predicted.shape[0] * scalar))
            shape = (int(p_img_predicted.shape[0] * scalar), int(p_img_predicted.shape[1] * scalar))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, shape)
            p_img_predicted = cv2.morphologyEx(p_img_predicted, cv2.MORPH_CLOSE, kernel=kernel)

            # larges component
            p_img_predicted = getLargestCC(p_img_predicted)
            p_img_predicted = p_img_predicted.astype(int)

            # fill holes
            p_img_predicted = ndimage.binary_fill_holes(p_img_predicted)
            p_img_predicted = p_img_predicted.astype(int)

            #plt.imshow(p_img_predicted)
            #plt.show()

            # normalize predicted image
            p_img_predicted = cv2.normalize(src=p_img_predicted, dst=None, alpha=0,
                                            beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            p_img_predicted = feature.canny(p_img_predicted)
            p_img_predicted = p_img_predicted.astype(int)

            p_img_predicted = cv2.normalize(src=p_img_predicted, dst=None, alpha=0,
                                            beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            plt.imshow(p_img_predicted)
            plt.show()

            plt.imshow(reference)
            plt.show()

        cv2.imwrite(f'{save_path}/{filename.stem}_pred_bin.png', p_img_predicted)
        cv2.imwrite(f'{save_path}/{filename.stem}_ref.png', reference)

        print(i)
        i = i + 1

        '''
        # create figure comparing reference and predicted mask
        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3)
        ax0.imshow(img, cmap='gray')
        ax0.title.set_text('SAR')
        ax0.axes.xaxis.set_visible(False)
        ax0.axes.yaxis.set_visible(False)
        ax1.imshow(reference, cmap='gray')
        ax1.title.set_text('Zone')
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        ax2.imshow(p_img_predicted, cmap='gray')
        ax2.title.set_text('Prediction')
        ax2.axes.xaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)
        if save_fig:
            plt.savefig(f'{save_path}/{i}_{filename.name}', bbox_inches='tight')
        plt.show()

        # calculate metrics
        report = metrics.classification_report(reference.flatten(), p_img_predicted.flatten(), output_dict=True)
        sensitivity.append(report['255']['recall'])
        specificity.append(report['0']['recall'])

        dice_score = metrics.f1_score(reference.flatten() / 255, p_img_predicted.flatten() / 255)
        iou_score = metrics.jaccard_score(reference.flatten() / 255, p_img_predicted.flatten() / 255)
        accuracy = metrics.accuracy_score(reference.flatten() / 255, p_img_predicted.flatten() / 255)

        # append metrics to array
        acc.append(accuracy)
        iou.append(iou_score)
        dice.append(dice_score)
        ref_list.append(reference)
        pred_list.append(p_img_predicted)
        '''

    '''
    tolerance = [11, 29, 67]
    dice = list()
    iou = list()

    for t in tolerance:

        ksize = t

        for i, _ in enumerate(ref_list):

            iou.append(relaxed_IoU(y_true=reference, y_pred=p_img_predicted, ksize=ksize))
            dice.append(relaxed_dice(y_true=reference / 255, y_pred=p_img_predicted / 255, ksize=ksize))

        print(ksize)
        print('Dice mean: {:.3f}, std: {:.3f}, median: {:.3f}'.format(np.mean(dice), np.std(dice), np.median(dice)))
        print('IOU mean: {:.3f}, std: {:.3f}, median: {:.3f}'.format(np.mean(iou), np.std(iou), np.median(iou)))
        print()

        np.save(f'{save_path}/iou_{ksize}.npy', iou)
        np.save(f'{save_path}/dice_{ksize}.npy', dice)


    data = {'accuracy': {'mean': np.mean(acc), 'std': np.std(acc)},
            'iou': {'mean': np.mean(iou), 'std': np.std(iou)},
            'dice': {'mean': np.mean(dice), 'std': np.std(dice)},
            'sensitivity': {'mean': np.mean(sensitivity), 'std': np.std(sensitivity)},
            'specificity': {'mean': np.mean(specificity), 'std': np.std(specificity)},
            'threshold': threshold}

    with open(f'{save_path}/evaluation.json', 'w') as json_file:
        json.dump(data, json_file)

    print('Accuracy:\t{:.3f}, dev:{:.3f}'.format(data['accuracy']['mean'], data['accuracy']['std']))
    print('IoU:\t{:.3f}, dev:{:.3f}'.format(data['iou']['mean'], data['iou']['std']))
    print('Dice:\t{:.3f}, dev:{:.3f}'.format(data['dice']['mean'], data['dice']['std']))
    print()
    print('Sensitivity:\t{:.3f}, dev:{:.3f}'.format(data['sensitivity']['mean'], data['sensitivity']['std']))
    print('Specificity:\t{:.3f}, dev:{:.3f}'.format(data['specificity']['mean'], data['specificity']['std']))
    print('Threshold:\t{:.3f}'.format(data['threshold']))

    if bar_plot:
        save_path = f'{out_path}/qualitative'
        make_path(save_path)

        # params
        width = 0.25
        r1 = np.arange(len(dice[:number]))
        r2 = [x + width for x in r1]
        r3 = [x + width for x in r2]

        # figure1
        fig, ax = plt.subplots()
        ax.bar(r1, acc[:number], color='b', width=width, label='accuracy')
        ax.bar(r2, iou[:number], color='g', width=width, label='iou')
        ax.bar(r3, dice[:number], color='r', width=width, label='dice')
        ax.set_ylabel('Scores')
        ax.set_title('Images')
        ax.set_xticks(r1)
        ax.legend()
        plt.ylim((0, 1))
        plt.plot()
        if save_fig:
            plt.savefig(f'{save_path}/acc_iou_dice.png')
        plt.show()

        # figure2
        fig, ax = plt.subplots()
        ax.bar(r1, sensitivity[:number], color='g', width=width, label='sensitivity')
        ax.bar(r2, specificity[:number], color='r', width=width, label='specificity')
        ax.set_ylabel('Scores')
        ax.set_title('Images')
        ax.set_xticks(r1)
        ax.legend()
        plt.ylim((0, 1))
        plt.plot()
        if save_fig:
            plt.savefig(f'{save_path}/sens_spec.png')
        plt.show()
        
        '''


qualitative_evaluation(save_fig=True)
