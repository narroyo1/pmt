"""
This module contains class EMDTest.
"""
# pylint: disable=no-member

import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from utils import to_tensor

# The number of points in X to center the test data partitions.
EMD_TEST_POINTS = 4

# This constant specifies the number of samples to be used on each sample partition.
# This number cannot be too big since EMD is very expensive to calculate.
# EMD_SAMPLES_PER_TEST_POINT = 600
EMD_SAMPLES_PER_TEST_POINT = 3000


class EMDTest:
    """
    This class implements the EMD (Earth Movers Distance) test on the model.
    """

    def __init__(self, x_test, y_test, device):
        self.y_test = y_test
        self.x_test = x_test

        self.x_test_pt = to_tensor(x_test, device)

        self.x_orderings_np = [
            np.argsort(self.x_test[:, i]) for i in range(self.x_test.shape[1])
        ]

    def calculate_emd(self, y_pred_d):
        """
        This function calculates the emd (Earth Mover's Distance) between a model prediction and
        the test data set. Calculating emd is very expensive (O(n^2)) so in order to speed up the
        calculation, the test and prediciton data points are separated into groups and their emd's
        are averaged together.
        """

        def emd(ordering):
            local_emd = np.zeros(EMD_TEST_POINTS)

            for point in range(EMD_TEST_POINTS):
                start = point * test_point_spacing
                stop = start + EMD_SAMPLES_PER_TEST_POINT

                # Calculate the distances between every point in one set and every point in
                # the other.
                distances = cdist(
                    self.y_test[ordering][start:stop],
                    y_pred_d[ordering][start:stop],
                )

                # Calculate the point to point matching the minimizes the EMD.
                assignment = linear_sum_assignment(distances)
                local_emd[point] = distances[assignment].sum() / (stop - start)

            mean_emd = np.mean(local_emd)

            return mean_emd

        num_data_points = self.y_test.shape[0]
        num_data_points_range = num_data_points - EMD_SAMPLES_PER_TEST_POINT

        if EMD_TEST_POINTS == 1:
            test_point_spacing = 0
        else:
            test_point_spacing = int(num_data_points_range / (EMD_TEST_POINTS - 1))

        num_dimensions = self.x_test.shape[1]
        if num_dimensions:
            mean_emds = np.zeros(num_dimensions)
            for dimension in range(num_dimensions):
                mean_emds[dimension] = emd(self.x_orderings_np[dimension])

            mean_emd = np.mean(mean_emds)
            return mean_emd

        return emd(np.random.permutation(self.y_test.shape[0]))

    def step(self, y_pred_d):
        """
        Runs and plots a step of the EMD test.
        """

        return self.calculate_emd(y_pred_d)
