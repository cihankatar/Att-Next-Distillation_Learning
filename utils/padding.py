"""Padding helpers for topological pseudo-mask generation."""

import numpy as np
from skimage.transform import resize


def adaptive_pad(image, pad=5, window=5):
    """Apply edge-aware binary padding to a two-dimensional image."""
    height, width = image.shape
    global_mean, global_std = image.mean(), image.std()
    output = np.zeros((height + 2 * pad, width + 2 * pad), dtype=image.dtype)
    output[pad : pad + height, pad : pad + width] = image
    decisions = []

    def decide(region):
        if np.all(region == 255):
            return 255
        return 0 if region.mean() > global_mean - global_std / 2 else 255

    for x in range(0, width, window):
        end = min(x + window, width)
        top = decide(image[:pad, x:end])
        bottom = decide(image[-pad:, x:end])
        decisions.extend((top == 255, bottom == 255))
        output[:pad, pad + x : pad + end] = top
        output[height + pad :, pad + x : pad + end] = bottom

    for y in range(0, height, window):
        end = min(y + window, height)
        left = decide(image[y:end, :pad])
        right = decide(image[y:end, -pad:])
        decisions.extend((left == 255, right == 255))
        output[pad + y : pad + end, :pad] = left
        output[pad + y : pad + end, width + pad :] = right

    return output if any(decisions) else image.copy()


def unpad_resize(padded_image, orig_shape, pad=1, target_size=256):
    """Remove scaled padding and restore the original image dimensions."""
    del target_size  # retained for compatibility with earlier calls
    height, width = orig_shape
    padded_height, padded_width = padded_image.shape
    top = int(padded_height * pad / (height + 2 * pad))
    bottom = int(padded_height * (height + pad) / (height + 2 * pad))
    left = int(padded_width * pad / (width + 2 * pad))
    right = int(padded_width * (width + pad) / (width + 2 * pad))
    cropped = padded_image[top:bottom, left:right]
    return resize(
        cropped,
        (height, width),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(padded_image.dtype)
