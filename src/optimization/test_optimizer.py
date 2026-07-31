import numpy as np

from src.optimization.optimizer import DeblurOptimizer

image = np.random.rand(260,346).astype(np.float32)

gradient = np.random.rand(260,346).astype(np.float32)

optimizer = DeblurOptimizer(
    blurry_image=image,
    event_prior=image,
    event_gradient=gradient,
)

latent, kernel = optimizer.optimize()

print(latent.shape)

print(kernel.shape)