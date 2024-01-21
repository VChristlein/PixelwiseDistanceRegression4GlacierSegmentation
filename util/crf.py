import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_bilateral
import numpy as np


def crf(img, probmap):
    H, W = probmap.shape

    probmap = np.tile(probmap[np.newaxis, :, :], (2, 1, 1))
    probmap[0, :, :] = 1 - probmap[1, :, :]

    U = unary_from_softmax(probmap)
    U = np.ascontiguousarray(U)

    NLABELS = 2

    # Create the pairwise bilateral term from the above image.
    # The two `s{dims,chan}` parameters are model hyper-parameters defining
    # the strength of the location and image content bilaterals, respectively.

    # pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01,), img=img, chdim=2)
    pairwise_energy = create_pairwise_bilateral(sdims=(512, 512), schan=(13,), img=img, chdim=2)

    # pairwise_energy now contains as many dimensions as the DenseCRF has features,
    # which in this case is 3: (x,y,channel1)
    # img_en = pairwise_energy.reshape((-1, H, W))  # Reshape just for plotting

    d = dcrf.DenseCRF2D(W, H, NLABELS)
    d.setUnaryEnergy(U)
    # d.addPairwiseGaussian(sxy=(21, 21), compat=1)
    d.addPairwiseEnergy(pairwise_energy, compat=3)  # `compat` is the "strength" of this potential.

    Q = d.inference(3)
    # d.klDivergence(Q) / (H*W)
    return np.argmax(Q, axis=0).reshape((H, W))
