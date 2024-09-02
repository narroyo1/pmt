import numpy as np
import torch

EPSILON = 0.00001


def _dot_product_batch(X, Y):
    view_size = X.shape[0] * Y.shape[1]
    # The dot product of a unit vector with itself is 1.0.
    result = torch.bmm(
        X.reshape(view_size, 1, X.shape[2]), Y.reshape(view_size, Y.shape[2], 1)
    )
    return result.view((X.shape[0], X.shape[1]))


class MovementScalarCalculator:
    def __init__(self, *, z_pins, device):
        self.z_pins = z_pins
        self.device = device

    def get_distances(self, D, z_pins):
        """
        Based on this algorithm
        https://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection

        D is the unit vector describing the z-direction.
        """
        a = _dot_product_batch(D, D)
        # a should always be 1.0
        ########################
        err = torch.abs(a - 1.0)
        if (err > EPSILON).any():
            assert False
        ########################
        # dimensions: (z-pins, data points, output dimensions)
        z_pins_offset = self.z_pins.z_offset
        O = z_pins - z_pins_offset
        # dimensions: (z-pins, data points, output dimensions)
        b = 2 * _dot_product_batch(O, D)
        # dimensions: (z-pins, data points, output dimensions)
        z_pins_radio = self.z_pins.z_radio
        c = _dot_product_batch(O, O) - z_pins_radio**2
        # dimensions: (z-pins, data points, output dimensions)
        discriminant = b * b - 4 * a * c

        # These are the indexes for which the z-line is inside Z.
        nidx = discriminant > 0.0

        # dimensions: (z-pins, data points, output dimensions)
        # Elements in t_pos will always be positive and elements in t_neg will always be
        # negative.
        t_pos = (-b + torch.sqrt(discriminant)) / (2 * a)
        ########################
        if (t_pos[nidx].isnan()).any():
            assert False
        ########################
        t_neg = (-b - torch.sqrt(discriminant)) / (2 * a)
        ########################
        if (t_neg[nidx].isnan()).any():
            assert False
        ########################
        if (t_pos[nidx] - t_neg[nidx] > z_pins_radio * 2.0 + EPSILON).any():
            assert False

        return t_pos, t_neg, nidx
