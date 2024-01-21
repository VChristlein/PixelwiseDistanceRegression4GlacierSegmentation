import cv2
import numpy as np
from skimage import util
from scipy.ndimage.morphology import distance_transform_edt
from skimage import morphology


def create_distance_map(mask_front, gamma=7):
    """
    creates a distance map of a given image mask front

    :param mask_front: image front
    :param gamma: parameter defining the
    :return: the distance map of given image
    """

    mask_front = mask_front / 255
    mask_front = util.invert(mask_front)
    distance_map = distance_transform_edt(mask_front)

    distance_map = (1 - distance_map / np.max(distance_map)) ** gamma
    distance_map = cv2.normalize(src=distance_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return distance_map


def distance_map_to_front(distance_map):
    def get_thres_value(image, thres=0.95, default_value=245):
        thres = image.size * thres

        hist = cv2.calcHist([distance_map], [0], None, [256], [0, 256])

        s = 0
        for idx, value in enumerate(hist):
            s += value

            if s > thres:
                return idx

        # if calculation fails return default value
        return default_value

    thres_value = get_thres_value(distance_map)

    _, distance_map_thres = cv2.threshold(distance_map, thres_value, 255, cv2.THRESH_TOZERO)

    mask_front = morphology.thin(distance_map_thres)

    return np.uint8(mask_front)
