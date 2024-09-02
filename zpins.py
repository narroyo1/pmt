"""
This module contains class ZPins.
"""
# pylint: disable=bad-continuation

import numpy as np

from utils import to_tensor


# Converted from https://baezortega.github.io/2018/10/14/hypersphere-sampling/
def _sample_hypersphere(num_samples, num_dimensions, surface=False):
    """
    Rejection-free sampling of num_samples random points (vectors of coordinates
    [x_1, ..., x_D]) in/on a D-dimensional sphere of radius r.
    """
    # Sample num_dimensions vectors of num_samples Gaussian coordinates
    # np.random.seed(1)
    samples = np.array(
        [np.random.normal(size=num_samples) for _ in range(num_dimensions)]
    ).T

    # Normalize all distances (radii) to 1
    radii = np.sqrt(np.sum(samples**2, axis=1))
    samples = samples / radii[:, None]

    # Sample num_samples radii with exponential distribution
    # (unless points are to be on the surface)
    if not surface:
        new_radii = np.power(np.random.uniform(size=num_samples), 1 / num_dimensions)
        samples = samples * new_radii[:, None]

    return samples


class ZPins:
    """
    This class encapsulates the z-pins space.
    """

    def __init__(self, *, z_pins_config, device):
        self.z_dimensions = z_pins_config["z_pins_dimensions"]
        self.z_radio = z_pins_config.get("z_pins_radio")
        self.device = device

        self.z_offset = to_tensor([0.0] * self.z_dimensions, self.device)

    def set_radio_and_offset(self, z_radio, z_offset):
        self.z_radio = self.z_radio or z_radio
        self.z_offset = z_offset

    def sample(self, size, scale=1.0):
        """
        This method gets "size" random elements in Z-space.
        """
        z_pins = _sample_hypersphere(size, self.z_dimensions)

        return to_tensor(z_pins, device=self.device) * self.z_radio + self.z_offset
