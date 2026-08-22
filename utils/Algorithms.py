"""Classical segmentation baselines used during pseudo-mask generation."""

import cv2
import numpy as np
import skimage.filters as filters
import skimage.morphology as morphology
import skimage.segmentation as segmentation


def _clean_mask(mask):
    mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=32 * 32)
    return morphology.remove_small_objects(mask, min_size=64).astype(np.uint8)


def morphological_chan_vese_segmentation(image):
    """Segment a grayscale image with morphological Chan-Vese."""
    image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_uint8)

    height, width = image_uint8.shape
    init_mask = np.zeros_like(image_uint8, dtype=bool)
    init_mask[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = True
    mask = segmentation.morphological_chan_vese(
        enhanced,
        200,
        init_level_set=init_mask,
    )
    return _clean_mask(mask), enhanced


def random_walker_pseudo_mask(image):
    """Create a binary pseudo-mask from a grayscale image using random walker."""
    image = np.asarray(image, dtype=np.float32)
    threshold = filters.threshold_otsu(image)
    delta = 0.10 * (image.max() - image.min())

    markers = np.zeros_like(image, dtype=np.int32)
    markers[image < threshold - delta] = 1
    markers[image > threshold + delta] = 2

    if not np.any(markers == 1) or not np.any(markers == 2):
        return _clean_mask(image < threshold)

    labels = segmentation.random_walker(image, markers, beta=90, mode="bf")
    return _clean_mask(labels == 1)
