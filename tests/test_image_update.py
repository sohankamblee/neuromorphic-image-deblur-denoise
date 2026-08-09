import numpy as np

from src.optimization.image_update import update_latent_image

image = np.random.rand(260,346).astype(np.float32)

kernel = np.zeros((25,25),dtype=np.float32)
kernel[12,12] = 1.0

event_prior = np.random.rand(260,346).astype(np.float32)

latent = update_latent_image(
    blurry_image=image,
    kernel=kernel,
    event_prior=event_prior,
    latent_image=image.copy(),
    alpha=1.0,
    beta=0.064,
    gamma=2.0,
)

print(latent.shape)
print(latent.dtype)
print(np.min(latent))
print(np.max(latent))