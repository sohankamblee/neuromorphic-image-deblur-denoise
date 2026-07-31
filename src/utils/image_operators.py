"""
image_operators.py

Low-level mathematical operators used throughout the optimization.

These operators correspond to the discrete image operators used in the
optimization formulation of the paper.

Author: Sohan Kamble
Project: Neuromorphic Image Deblur & Denoise
"""

from __future__ import annotations

import cv2
import numpy as np


# ==========================================================
# Image Gradient
# ==========================================================

def gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute forward finite-difference gradients.

    Parameters
    ----------
    image : ndarray (H,W)

    Returns
    -------
    gx : ndarray
    gy : ndarray
    """

    if image.ndim != 2:
        raise ValueError("gradient() expects a grayscale image.")

    image = image.astype(np.float32)

    gx = np.zeros_like(image)
    gy = np.zeros_like(image)

    gx[:, :-1] = image[:, 1:] - image[:, :-1]
    gy[:-1, :] = image[1:, :] - image[:-1, :]

    return gx, gy


# ==========================================================
# Divergence
# ==========================================================

def divergence(
    gx: np.ndarray,
    gy: np.ndarray,
) -> np.ndarray:
    """
    Compute divergence operator.

    This is the adjoint of the forward gradient operator.

    Parameters
    ----------
    gx
        Horizontal gradient.

    gy
        Vertical gradient.

    Returns
    -------
    div : ndarray
    """

    if gx.shape != gy.shape:
        raise ValueError("gx and gy must have identical shape.")

    h, w = gx.shape

    div = np.zeros((h, w), dtype=np.float32)

    # horizontal

    div[:, 0] = gx[:, 0]

    div[:, 1:-1] = gx[:, 1:-1] - gx[:, :-2]

    div[:, -1] = -gx[:, -2]

    # vertical

    div[0, :] += gy[0, :]

    div[1:-1, :] += gy[1:-1, :] - gy[:-2, :]

    div[-1, :] += -gy[-2, :]

    return div


# ==========================================================
# Image Convolution
# ==========================================================

def convolve(
    image: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    2D convolution using OpenCV.
    """

    return cv2.filter2D(
        image.astype(np.float32),
        ddepth=-1,
        kernel=kernel.astype(np.float32),
        borderType=cv2.BORDER_REFLECT,
    )


# ==========================================================
# Kernel Flip
# ==========================================================

def flip_kernel(
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Rotate kernel by 180 degrees.

    MATLAB equivalent:

        rot90(kernel,2)
    """

    return np.flip(kernel)


# ==========================================================
# Gradient Magnitude
# ==========================================================

def gradient_magnitude(
    gx: np.ndarray,
    gy: np.ndarray,
) -> np.ndarray:
    """
    Euclidean gradient magnitude.
    """

    return np.sqrt(gx * gx + gy * gy)


# ==========================================================
# Normalize Image
# ==========================================================

def normalize(
    image: np.ndarray,
) -> np.ndarray:
    """
    Normalize image to [0,1].
    """

    image = image.astype(np.float32)

    mn = image.min()
    mx = image.max()

    if mx - mn < 1e-8:
        return np.zeros_like(image)

    return (image - mn) / (mx - mn)