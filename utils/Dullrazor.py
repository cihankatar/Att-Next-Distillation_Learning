"""Lightweight DullRazor-style hair removal for CHW image tensors."""

import torch
import torch.nn.functional as F


def blackhat_transform(gray_tensor, kernel_size=9):
    padding = kernel_size // 2
    dilated = F.max_pool2d(gray_tensor, kernel_size, 1, padding)
    closed = -F.max_pool2d(-dilated, kernel_size, 1, padding)
    return (closed - gray_tensor).clamp(0, 1)


def patch_fill(image, mask, kernel_size=15):
    padding = kernel_size // 2
    masked_input = image * (1 - mask)
    norm = F.avg_pool2d(1 - mask, kernel_size, 1, padding) + 1e-8
    smooth = F.avg_pool2d(masked_input, kernel_size, 1, padding) / norm
    return masked_input + smooth * mask


def dullrazor(image, threshold=0.05):
    """Remove thin dark structures from a ``[C, H, W]`` image in ``[0, 1]``."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("dullrazor expects an RGB tensor shaped [3, H, W]")

    image = image.unsqueeze(0)
    gray = (
        0.2989 * image[:, 0]
        + 0.5870 * image[:, 1]
        + 0.1140 * image[:, 2]
    ).unsqueeze(1)
    hair_mask = (blackhat_transform(gray) > threshold).float().repeat(1, 3, 1, 1)
    return patch_fill(image, hair_mask).squeeze(0).clamp(0, 1)
