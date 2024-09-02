"""
This module contains class Tester.
"""

import numpy as np

from emd_test import EMDTest

from utils import to_tensor

# pylint: disable=bad-continuation


class Tester:
    """
    This class implements a mechanism to test the performance of a neural network
    that produces stochastic outputs.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        *,
        experiment,
        x_test,
        y_test,
        y_train,
        z_pins,
        plotter,
        writer,
        device,
    ):
        self.plotter = plotter
        self.writer = writer
        self.device = device

        self.z_pins = z_pins
        self.x_test_pt = to_tensor(x_test, device)
        self.y_test_pt = to_tensor(y_test, device)
        self.x_test = x_test
        self.y_test = y_test

        self.skip_epochs = experiment["skip_epochs"]

        if experiment.get("emd_test", True):
            self.emd_test = EMDTest(x_test, y_test, device)
        else:
            self.emd_test = None

        self.emds = []
        self.z_test_pt = None
        self.y_train = y_train
        self.base_emds = []

    def step(
        self,
        model,
        epoch,
    ):
        """
        Runs the tests on the model.
        """
        # Only run tests every number of epochs.
        if epoch % self.skip_epochs != 0:
            return

        if self.z_test_pt is None:
            test_size = self.x_test_pt.shape[0]
            self.z_test_pt = self.z_pins.sample(test_size)

        import torch

        with torch.no_grad():
            y_sample = model.forward_z(self.x_test_pt, self.z_test_pt)

        # Create a numpy version of the prediction tensor.
        y_pred_d = y_sample.cpu().detach().numpy()
        if self.emd_test:
            mean_emd = self.emd_test.step(y_pred_d)
            y_base = self.y_train[
                np.random.choice(
                    self.y_train.shape[0], self.y_test.shape[0], replace=False
                )
            ]
            base_emd = self.emd_test.step(y_base)

            self.writer.log_emd(mean_emd, base_emd, epoch)
            self.emds.append(mean_emd)
            self.base_emds.append(base_emd)

        self.plotter.plot_samples(y_pred_d, model, y_sample, epoch)
