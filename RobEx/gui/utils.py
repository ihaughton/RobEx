import numpy as np
import cv2
from nptyping import NDArray, UInt8
from typing import Any

def visualise_segmentation(im: NDArray[(Any, ...), UInt8], masks: NDArray[(Any, ...), UInt8], colourmap: NDArray[(Any, ...), UInt8]) -> NDArray[(Any, ...), UInt8]:
    """
    Visualize segmentations nicely.
    :param im: a [H x W x 3] RGB image. numpy array of dtype np.uint8
    :param masks: a [H x W] numpy array of dtype np.uint8 with values in {0, ..., nc-1}
    :param colourmap: num_classes x 3 colourmap

    :return: a [H x W x 3] numpy array of dtype np.uint8
    """

    masks = masks.astype(int)
    im = im.copy()

    # Mask
    imgMask = np.zeros(im.shape)

    # Draw color masks
    for i in np.unique(masks):

        # Get the color mask
        color_mask = np.array(colourmap[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Add to the mask
        imgMask[e] = color_mask

    # Add the mask to the image
    imgMask = (imgMask * 255).round().astype(np.uint8)
    im = cv2.addWeighted(im, 0.5, imgMask, 0.5, 0.0)

    return im
