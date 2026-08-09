"""
kernel_update.py

Implements blur kernel estimation
(Eq. 22–23).
"""

from __future__ import annotations

import numpy as np

from src.utils.image_operators import (
    gradient,
    fft2,
    ifft2,
)


def update_kernel(
    latent_image: np.ndarray,
    blurry_image: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Estimate blur kernel using Eq. (23).

    Parameters
    ----------
    latent_image
        Current estimate of S.

    blurry_image
        Observed blurry image B.

    sigma
        Gaussian regularization weight.

    Returns
    -------
    kernel
    """

    # ----------------------------------
    # Compute gradients
    # ----------------------------------

    sx, sy = gradient(latent_image)

    bx, by = gradient(blurry_image)

    # ----------------------------------
    # FFT
    # ----------------------------------

    Fs_x = fft2(sx)
    Fs_y = fft2(sy)

    Fb_x = fft2(bx)
    Fb_y = fft2(by)

    # ----------------------------------
    # Eq. (23)
    # ----------------------------------

    numerator = (
        np.conj(Fs_x) * Fb_x
        +
        np.conj(Fs_y) * Fb_y
    )

    denominator = (
        np.abs(Fs_x) ** 2
        +
        np.abs(Fs_y) ** 2
        +
        sigma
    )

    kernel = np.real(
        ifft2(
            numerator / (denominator + 1e-8)
        )
    )

    # ----------------------------------
    # Post-processing
    # ----------------------------------

    kernel[kernel < 0] = 0

    s = kernel.sum()

    if s > 1e-8:
        kernel /= s

    return kernel.astype(np.float32)


def crop_kernel(
    kernel: np.ndarray,
    kernel_size: int = 25,
) -> np.ndarray:
    """
    Extract the central blur kernel from the
    full-size FFT solution.

    Parameters
    ----------
    kernel
        Full-size kernel returned by update_kernel().

    kernel_size
        Desired PSF size.

    Returns
    -------
    Cropped kernel.
    """

    H, W = kernel.shape

    cy = H // 2
    cx = W // 2

    r = kernel_size // 2

    cropped = kernel[
        cy-r:cy+r+1,
        cx-r:cx+r+1,
    ].copy()

    cropped[cropped < 0] = 0

    s = cropped.sum()

    if s > 1e-8:
        cropped /= s

    return cropped.astype(np.float32)