import numpy as np

from src.optimization.kernel_update import update_kernel
from src.optimization.kernel_update import (
    update_kernel,
    crop_kernel,
)

image = np.random.rand(260,346).astype(np.float32)

full_kernel = update_kernel(
    latent_image=image,
    blurry_image=image,
    sigma=2.0,
)

kernel = crop_kernel(
    full_kernel,
    kernel_size=25,
)
print("Full kernel :", full_kernel.shape)

print("Kernel :", kernel.shape)

print("Kernel sum :", kernel.sum())

print("Kernel min :", kernel.min())

print("Kernel max :", kernel.max())