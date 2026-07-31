"""
Main optimization loop corresponding to Algorithm 1.
"""

import numpy as np

from .latent_update import update_latent_image
from .kernel_update import update_kernel
from .auxiliary_update import update_auxiliary


class DeblurOptimizer:

    def __init__(
        self,
        blurry_image,
        event_prior,
        event_gradient,
        kernel_size=25,
        max_iterations=10,
    ):

        self.B = blurry_image

        self.event_prior = event_prior

        self.event_gradient = event_gradient

        self.kernel_size = kernel_size

        self.max_iterations = max_iterations

        self.latent = None

        self.kernel = None

    def initialize(self):

        self.latent = self.B.copy()

        self.kernel = np.zeros(
            (self.kernel_size, self.kernel_size),
            dtype=np.float32,
        )

        center = self.kernel_size // 2

        self.kernel[center, center] = 1.0

    def optimize(self):

        self.initialize()

        for iteration in range(self.max_iterations):

            print(f"Iteration {iteration+1}")

            self.latent = update_latent_image(
                self.latent,
                self.kernel,
                self.event_gradient,
            )

            auxiliary = update_auxiliary(
                self.latent,
            )

            self.kernel = update_kernel(
                self.latent,
                auxiliary,
                self.kernel,
            )

        return self.latent, self.kernel