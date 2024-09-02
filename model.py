"""
This module contains class Model.
"""
# pylint: disable=bad-continuation

import torch
import torch.nn as nn


def get_z_pin_preds_specific(model, x_pt, z_pins):
    """
    This method evaluates the model over every point in x on the provided z_pins.
    """
    assert x_pt.shape[0] == z_pins.shape[1]
    z_pins_size = z_pins.shape[0]

    # Create a tensor with a copy of the elements in the batch for every z sample. This
    # will be matched with z_pins_ex when finding targets.
    # [x0, -> for z sample 0
    #  x1, -> for z sample 0
    #  ...
    #  xn, -> for z sample 0
    #  x0, -> for z sample 1
    #  x1, -> for z sample 1
    #  ...
    #  xn, -> for z sample 1
    #  ...
    #  x0, -> for z sample S
    #  x1, -> for z sample S
    #  ...
    #  xn, -> for z sample S]
    # dimensions: (data points * z-pins, input dimensions)
    x_ex = torch.cat(z_pins_size * [x_pt]).to(device=model.device)

    z_pins_ex = z_pins.view(z_pins.shape[0] * z_pins.shape[1], z_pins.shape[2])

    return _get_preds(model, x_ex, z_pins_ex, x_pt.shape[0], z_pins_size)


def _get_preds(model, x_ex, z_pins_ex, input_size, z_pins_size):
    # Turn off grad while we get our predictions.
    with torch.no_grad():
        # Run the model with all the elements x on every z sample.
        # [y <- x0 z0,
        #  y <- x1 z0,
        #  ...
        #  y <- xn z0,
        #  y <- x0 z1,
        #  y <- x1 z1,
        #  ...
        #  y <- xn z1
        #  ...
        #  y <- x0 zS,
        #  y <- x1 zS,
        #  ...
        #  y <- xn zS]
        # dimensions: (data points * z-pins, output dimensions)
        y_predict = model.forward_z(x_ex, z_pins_ex)

        # Create a matrix view of the results with a column for every element and a row for
        # every z sample.
        # [[y <- x0 z0, y <- x1 z0, ..., y <- xn z0],
        #  [y <- x0 z1, y <- x1 z1, ..., y <- xn z1],
        #  ...,
        #  [y <- x0 zS, y <- x1 zS, ..., y <- xn zS]]
        # dimensions: (z-pins, data points, output dimensions)
        # y_predict_mat = y_predict.view(z_pins_size, input_size, y_predict.shape[1])
        y_predict_mat = y_predict.view(z_pins_size, input_size, -1)

        return y_predict_mat


class LayersBuilder:
    def __init__(self, *, hidden_layers, hidden_size, function):
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.function = self._get_function(function)

    @staticmethod
    def _get_function(function):
        if function == "tanh":
            return nn.Tanh()
        if function == "lrelu":
            return nn.LeakyReLU()  # 0.1)
        assert False

    def build_network(self, *, input_size, x_space_size, output_size, device):
        return self._build_layers(
            input_size=input_size,
            x_space_size=x_space_size,
            output_size=output_size,
            device=device,
        )

    def _build_layers(self, *, input_size, x_space_size, output_size, device):
        # keep_prob = 0.95
        layers = []
        layers.append(
            nn.Linear(x_space_size + input_size, self.hidden_size, device=device)
        )
        # layers.append(nn.BatchNorm1d(self.hidden_size))
        layers.append(self.function)
        # layers.append(nn.Dropout(1-keep_prob))

        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size, device=device))
            # layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(self.function)
            # layers.append(nn.Dropout(1-keep_prob))

        layers.append(nn.Linear(self.hidden_size, output_size, device=device))

        return layers


class Model(nn.Module):
    """
    This is the neural network model.
    """

    def __init__(
        self,
        *,
        z_space_size,
        x_space_size,
        y_space_size,
        device,
        layers_builder,
    ):
        super().__init__()
        self.device = device

        layers = layers_builder.build_network(
            input_size=z_space_size,
            x_space_size=x_space_size,
            output_size=y_space_size,
            device=self.device,
        )
        self.net = nn.Sequential(*layers).to(self.device)

        layers = layers_builder.build_network(
            input_size=y_space_size,
            x_space_size=x_space_size,
            output_size=z_space_size,
            device=self.device,
        )
        self.invnet = nn.Sequential(*layers).to(self.device)

    def forward_y(self, x_pt, y_pt):
        # If there is data on x merge it with y as an input
        if x_pt.shape[1] > 0:
            mixed_pt = torch.cat(
                (x_pt.view(x_pt.size(0), -1), y_pt.view(y_pt.size(0), -1)), dim=1
            )
        else:
            mixed_pt = y_pt
        z_pt = self.invnet(mixed_pt)

        return z_pt

    def forward_z(self, x_pt, z_pt):
        """
        This method runs a forward pass through the model with the provided input x
        and z-pins.
        """

        mixed_pt = torch.cat(
            (x_pt.view(x_pt.size(0), -1), z_pt.view(z_pt.size(0), -1)), dim=1
        )

        return self.forward(mixed_pt)

    def forward(self, x_pt):
        """
        This method runs a forward pass through the model with the provided input x.
        """
        return self.net(x_pt)
