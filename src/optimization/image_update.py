import numpy as np

from src.utils.image_operators import (
    fft2,
    ifft2,
    psf_to_otf,
)

from src.optimization.auxiliary_update import update_auxiliary

def update_latent_image(
    blurry_image: np.ndarray,
    kernel: np.ndarray,
    event_prior: np.ndarray,
    latent_image: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    shape = blurry_image.shape

    K = psf_to_otf(kernel, shape)

    B = fft2(blurry_image)

    I_tau = fft2(event_prior)

    dx = np.array([[1, -1]], dtype=np.float32)
    dy = np.array([[1], [-1]], dtype=np.float32)

    Dx = psf_to_otf(dx, shape)
    Dy = psf_to_otf(dy, shape)

    z_h, z_v = update_auxiliary(
        latent_image,
        beta,
        gamma,
    )

    Zh = fft2(z_h)
    Zv = fft2(z_v)

    numerator = (
        np.conj(K) * B
        + alpha * (
            np.conj(Dx) + np.conj(Dy)
        ) * I_tau
        + gamma * (
            np.conj(Dx) * Zh
            + np.conj(Dy) * Zv
        )
    )

    denominator = (
        np.abs(K) ** 2
        + (alpha + gamma)
        * (
            np.abs(Dx) ** 2
            + np.abs(Dy) ** 2
        )
    )

    denominator += 1e-8

    latent = np.real(
        ifft2(
            numerator / denominator
        )
    )

    return latent.astype(np.float32)