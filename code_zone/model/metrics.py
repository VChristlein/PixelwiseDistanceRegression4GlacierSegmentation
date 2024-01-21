from keras import backend as K


def iou_coef(y_true, y_pred, smooth=1., axis=[1, 2, 3]):
    '''
    intersection = K.sum(y_true * y_pred, axis=axis)
    union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return (intersection + smooth) / (union + smooth)
    '''

    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)

    intersection = K.sum(y_true * y_pred, axis=1, keepdims=True) + smooth
    union = K.sum(y_true, axis=1, keepdims=True) + K.sum(y_pred, axis=1, keepdims=True) + smooth
    return K.mean(intersection / union)


def dice_coef(y_true, y_pred, smooth=1., axis=[1, 2, 3]):
    '''
    intersection = K.sum(y_true * y_pred, axis=axis)
    union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2 * intersection + smooth) / (union + smooth)
    '''

    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)
    intersection = 2. * K.sum(y_true * y_pred, axis=1, keepdims=True) + smooth
    union = K.sum(y_true, axis=1, keepdims=True) + K.sum(y_pred, axis=1, keepdims=True) + smooth

    return K.mean(intersection / union)
