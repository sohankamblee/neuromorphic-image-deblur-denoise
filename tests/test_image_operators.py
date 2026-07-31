import numpy as np

from src.utils.image_operators import (
    gradient,
    divergence,
    convolve,
    flip_kernel,
    gradient_magnitude,
)


image = np.random.rand(260,346).astype(np.float32)

gx, gy = gradient(image)

print("Gradient X:", gx.shape)
print("Gradient Y:", gy.shape)

div = divergence(gx, gy)

print("Divergence:", div.shape)

kernel = np.ones((3,3), dtype=np.float32) / 9

conv = convolve(image, kernel)

print("Convolution:", conv.shape)

flip = flip_kernel(kernel)

print("Kernel:", flip.shape)

mag = gradient_magnitude(gx, gy)

print("Magnitude:", mag.shape)