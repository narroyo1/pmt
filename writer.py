"""
This module contains class Writer.
"""


import pickle

from torch.utils.tensorboard import SummaryWriter

SKIP_ITERATIONS = 50


class Writer:
    """
    This class logs training information that can be viewed with tensorboard.
    """

    def __init__(self):
        self.summwriter = SummaryWriter()

        self.loss_y = []
        self.emd = []
        self.base_emd = []

        file = open("values", "wb")

    def save(self):
        with open("values", "ab") as file:
            pickle.dump(
                {"loss_y": self.loss_y, "emd": self.emd, "base_emd": self.base_emd},
                file,
            )
        self.loss_y = []
        self.emd = []
        self.base_emd = []

    def log_loss_y(self, loss_mse, iteration):
        if iteration % SKIP_ITERATIONS == 0:
            self.summwriter.add_scalar("loss_y", loss_mse, iteration)
        self.loss_y.append(loss_mse)

    def log_loss_z(self, loss_mse, iteration):
        if iteration % SKIP_ITERATIONS == 0:
            self.summwriter.add_scalar("loss_z", loss_mse, iteration)

    def log_loss_p(self, loss_mae, iteration):
        if iteration % SKIP_ITERATIONS == 0:
            self.summwriter.add_scalar("loss_p", loss_mae, iteration)

    def log_out_ratio(self, out_ratio, iteration):
        if iteration % SKIP_ITERATIONS == 0:
            self.summwriter.add_scalar("out_ratio", out_ratio, iteration)

    def log_emd(self, emd, base_emd, epoch):
        """
        This method logs the emd (Earth Mover's Distance) scalar.
        """
        scalars = {"emd": emd, "base_emd": base_emd}
        self.summwriter.add_scalars("emds", scalars, epoch)
        self.emd.append(emd)
        self.base_emd.append(base_emd)
