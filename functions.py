"""
This module can be used to compose a stochastic function and train a neural network
to aproximate it.
"""

import numpy as np

from scipy.stats import truncnorm

# %%


###########################################
# 2D base functions
###########################################


def fn_rectangle(x_np, y_np, side1=9.0, side2=9.0):
    # Rectangle
    y_np = np.random.rand(x_np.shape[0], 2) * np.array([side1, side2])
    y_np[:, 0] -= side1 * 0.5
    y_np[:, 1] -= side2 * 0.5

    return x_np, y_np


###########################################
# noise functions
###########################################


def fn_rescale(x_np, y_np):
    rescale = np.random.choice(a=[1.0, 0.3], size=(x_np.shape[0],), p=[0.6, 0.4])

    y_np -= 5
    y_np *= rescale[..., np.newaxis]

    return x_np, y_np


def binder(func, x_space_size=1, y_space_size=1, *args, **kwargs):
    """
    This function binds named arguments to a function taking 1 numpy array positional argument.
    """

    def helper(x_np, y_np):
        return func(x_np, y_np, *args, **kwargs)

    helper.x_space_size = x_space_size
    helper.y_space_size = y_space_size
    helper.name = func.__name__

    return helper
