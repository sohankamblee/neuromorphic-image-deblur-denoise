"""
Auxiliary variable update.

Implements Eq. (21) of the paper.
"""

from __future__ import annotations

import numpy as np

from src.utils.image_operators import gradient


def update_auxiliary(
    latent_image: np.ndarray,
    beta: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute auxiliary variables (zh, zv).

    Implements Eq. (21).

    Parameters
    ----------
    latent_image : ndarray
        Current latent image S.

    beta : float

    gamma : float

    Returns
    -------
    zh : ndarray
    zv : ndarray
    """

    gx, gy = gradient(latent_image)

    threshold = beta / gamma

    magnitude_sq = gx**2 + gy**2

    mask = magnitude_sq > threshold

    zh = np.where(mask, gx, 0.0)

    zv = np.where(mask, gy, 0.0)

    return zh.astype(np.float32), zv.astype(np.float32)