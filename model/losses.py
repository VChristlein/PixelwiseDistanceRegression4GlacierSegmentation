import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from skimage import util
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
from tensorflow.keras.losses import *


def dice_loss(y_true, y_pred, smooth=1.):
    # clipping to tackle infs
    # epsilon = K.epsilon()
    # y_true = K.clip(y_true, epsilon, 1. - K.epsilon())
    # y_pred = K.clip(y_pred, epsilon, 1. - K.epsilon())

    # flatten label and prediction tensors
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = 2 * K.sum(y_pred * y_true)
    return 1. - (intersection + smooth) / (K.sum(y_pred) + K.sum(y_true) + smooth + K.epsilon())


def cross_dice(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


# source: https://github.com/umbertogriffo/focal-loss-keras
def focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def distance_loss(gamma=7):
    def loss_function(y_true, y_pred):
        y_pred = util.invert(y_pred)
        y_pred = distance_transform_edt(y_pred)
        y_pred = (1 - y_pred / np.max(y_pred)) ** gamma

        y_true = util.invert(y_true)
        y_true = distance_transform_edt(y_true)
        y_true = (1 - y_true / np.max(y_true)) ** gamma

        return mean_squared_error(y_true, y_pred)


def distance_weighted_dice_loss(y_true, y_pred, smooth=1):
    weight = 8.0
    dmaps = tf.map_fn(fn=lambda x: tf.cast(tf.py_function(distance_transform_edt, [1.0 - x[:, :, 0]], Tout=np.float64), tf.float32),
                      elems=y_true)  # EDT online
    dmaps = K.sigmoid(dmaps / weight)

    dmaps = (dmaps - 0.5) * 2.0 + y_true[:, :, :, 0]  # rescale from 0.5 - 1.0 to 0.0 to 1.0 + weights 1.0 for gt mask

    w_pred = y_pred[:, :, :, 0] * dmaps  # weighted prediction

    # calculate non binary dice loss
    intersection = K.sum(y_true[:, :, :, 0] * w_pred, axis=[1, 2])
    union = K.sum(y_true[:, :, :, 0], axis=[1, 2]) + K.sum(w_pred, axis=[1, 2])
    return 1 - K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def distance_weighted_bce_loss(y_true, y_pred):
    weight = 8.0
    dmaps = tf.map_fn(fn=lambda x: tf.cast(tf.py_function(distance_transform_edt, [1.0 - x[:, :, 0]], Tout=np.float64), tf.float32),
                      elems=y_true)  # EDT online
    dmaps = K.sigmoid(dmaps / weight)

    dmaps = (dmaps - 0.5) * 2.0 + y_true[:, :, :, 0]  # rescale from 0.5 - 1.0 to 0.0 to 1.0 + weights 1.0 for gt mask

    w_pred = y_pred[:, :, :, 0] * dmaps  # weighted prediction

    bce = BinaryCrossentropy()
    return bce(y_true=y_true, y_pred=w_pred)


def mcc_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, tf.float32))
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), tf.float32))
    fp = K.sum(K.cast((1 - y_true) * y_pred, tf.float32))
    fn = K.sum(K.cast(y_true * (1 - y_pred), tf.float32))

    up = tp * tn - fp * fn
    down = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    mcc = up / (down + K.epsilon())

    return 1 - K.mean(mcc)
