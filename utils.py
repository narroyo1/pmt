"""
This module contains misc utility functions.
"""


import numpy as np
import torch

EPSILON = np.finfo(np.float32).eps * 1000


def _assert_unit_vector(unit_vector):
    last_dimension = len(unit_vector.shape) - 1
    squared = unit_vector**2
    summation = torch.sum(squared, dim=last_dimension)
    length = torch.sqrt(summation)

    err = torch.abs(length - 1.0)
    if (err >= EPSILON).any():
        assert False


def get_unit_vector_and_magnitude(difference):
    last_dimension = len(difference.shape) - 1
    # dimensions: (z-pins, data points, output dimensions)
    squared = difference**2
    # dimensions: (z-pins, data points)
    summation = torch.sum(squared, dim=last_dimension)
    # dimensions: (z-pins, data points)
    length = torch.sqrt(summation)
    # dimensions: (z-pins, data points, output dimensions)
    D = difference / length.unsqueeze(last_dimension)
    _assert_unit_vector(D)
    D[D.isnan()] = 0.0

    return D, length


def sample_random(ranges, size):
    """
    This function creates a random numpy array of values within the provided ranges. It creates
    a uniformly distributed hypercube.
    @return nparray with shape (size, dims)
    """
    dims = ranges.shape[0]
    start = ranges[:, 0]
    end = ranges[:, 1]
    samples = (np.random.rand(size, dims) * (end - start)) + start

    return samples


def to_tensor(nparray, device):
    """
    This fuction transforms a numpy array into a tensorflow tensor.
    """
    return torch.tensor(nparray, dtype=torch.float32).to(device=device)
