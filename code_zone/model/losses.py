import tensorflow as tf
from keras import backend as K


def dice_loss(y_true, y_pred, smooth=1.):
    # clipping to tackle infs
    #epsilon = K.epsilon()
    #y_true = K.clip(y_true, epsilon, 1. - K.epsilon())
    #y_pred = K.clip(y_pred, epsilon, 1. - K.epsilon())

    # flatten label and prediction tensors
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = 2 * K.sum(y_pred * y_true)
    return 1. - (intersection + smooth) / (K.sum(y_pred) + K.sum(y_true) + smooth + K.epsilon())


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
