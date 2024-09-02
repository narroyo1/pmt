
import os

import numpy as np
import cv2

from matplotlib import pyplot

from utils import to_tensor


def _save_images(
    images, *, name="result.jpg", img_h=28, img_w=28, n_img_x=20, n_img_y=15
):
    import imageio

    images = images[: n_img_x * n_img_y].cpu().detach().numpy()
    images = images.reshape(n_img_x * n_img_y, img_h, img_w)
    cv2.imwrite("plots/" + name, _merge(images, [n_img_y, n_img_x]))
    # imageio.imwrite("plots/" + name, _merge(images, [n_img_y, n_img_x]))


def _merge(images, size):
    resize_factor = 1.0
    h, w = images.shape[1], images.shape[2]

    h_ = int(h * resize_factor)
    w_ = int(w * resize_factor)

    img = np.zeros((h_ * size[0], w_ * size[1]))

    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])

        image_ = cv2.resize(image, dsize=(w_, h_))  # , interp="bicubic")

        img[j * h_ : j * h_ + h_, i * w_ : i * w_ + w_] = image_

    return img * 255


class SampleRender:
    def __init__(self, *, datasets, device):
        self.x_test_pt = to_tensor(datasets.x_test, device)
        self.y_test_pt = to_tensor(datasets.y_test, device)

    def plot_test_data(self):
        pass

    def plot_samples(self, y_pred_d, model, y_sample, epoch):
        import torch

        with torch.no_grad():
            z_pt = model.forward_y(self.x_test_pt, self.y_test_pt)
            y_pt = model.forward_z(self.x_test_pt, z_pt)
        #self.plotter.plot_z_pins(z_pt.cpu().detach().numpy())

        self.step(epoch, z_pt, y_pt, y_sample, self.y_test_pt)

    def step(self, epoch, z_pt, y_pt, y_samples, y_test):
        _save_images(y_samples, name=f"result_{epoch}.jpg")
        _save_images(y_pt, name=f"reconstruction_{epoch}.jpg", n_img_x=10)
        _save_images(y_test, name=f"reconstruction_original.jpg", n_img_x=10)
        import matplotlib.pyplot as plt

        plt.figure(700, figsize=(40, 30))
        plt.clf()


FONT_SIZE = 30


class Plotter:
    """
    This class creates plots to track the model progress.
    """

    def __init__(self, *, datasets):
        self.x_test = datasets.x_test
        self.y_test = datasets.y_test
        self.angle = 0

    def plot_test_data(self):
        ANGLE_DIFF = 5

        num_y_dimensions = self.y_test.shape[1]
        if num_y_dimensions == 3:
            if not os.path.exists("plots/test_data"):
                os.makedirs("plots/test_data")

            for angle in range(0, 360, ANGLE_DIFF):
                fig = pyplot.figure(figsize=(20, 16), dpi=80)
                title = "test data"
                pyplot.suptitle(title, fontsize=FONT_SIZE)
                ax = fig.add_subplot(projection="3d")
                surf = ax.scatter(
                    self.y_test[:, 0],
                    self.y_test[:, 1],
                    self.y_test[:, 2],
                    s=0.8
                    # linewidth=0,
                    # antialiased=False,
                )
                ax.view_init(elev=30, azim=angle)
                pyplot.savefig(
                    f"plots/test_data/img_{angle:04}.png", bbox_inches="tight"
                )
        else:
            width = 20
            height = 16
            # train_s = 0.2  # self.options.get("train_s", 0.5)

            figure = pyplot.figure(0)
            figure.clf()

            figure.set_size_inches(width, height)
            title = "test data"
            pyplot.suptitle(title, fontsize=FONT_SIZE)
            pyplot.scatter(
                self.y_test[:, 0],
                self.y_test[:, 1],
                marker="x",
                s=1.5,
            )

            pyplot.grid()
            if not os.path.exists("plots"):
                os.makedirs("plots")
            figure.savefig(f"plots/test.png", bbox_inches="tight")

    def plot_samples(self, y_pred_d, model, y_sample, epoch):
        """
        This method plots the test dataset along with random samples.
        """
        ""
        dimensions = y_pred_d.shape[1]
        if dimensions == 3:
            fig = pyplot.figure(figsize=(20, 16), dpi=80)
            title = f"epoch {epoch}"
            pyplot.suptitle(title, fontsize=FONT_SIZE)
            ax = fig.add_subplot(projection="3d")
            # pyplot.figure(figsize=(20, 16), dpi=80)
            surf = ax.scatter(
                y_pred_d[:, 0],
                y_pred_d[:, 1],
                y_pred_d[:, 2],
                s=0.8
                # linewidth=0,
                # antialiased=False,
            )
            ax.view_init(elev=30, azim=self.angle)
            pyplot.savefig(f"plots/img_{epoch:04}.png", bbox_inches="tight")
            self.angle += 5

        elif dimensions == 2:
            pyplot.figure(figsize=(20, 16), dpi=80)
            title = f"epoch {epoch}"
            pyplot.suptitle(title, fontsize=FONT_SIZE)
            pyplot.scatter(
                y_pred_d[:, 0],
                y_pred_d[:, 1],
                marker="x",
                s=1.5,
            )

            pyplot.grid()

            if not os.path.exists("plots"):
                os.makedirs("plots")

            pyplot.savefig(f"plots/img_{epoch:04}.png", bbox_inches="tight")
