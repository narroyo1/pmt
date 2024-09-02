"""
This module contains class Trainer.
"""

import torch

from torch import optim

from model import get_z_pin_preds_specific
from movement_scalar_calculator import MovementScalarCalculator
from utils import get_unit_vector_and_magnitude


# pylint: disable=bad-continuation, not-callable

EPSILON = 0.00001


class PreTrainer:
    """
    This class implements a trainer class that runs before PMT.
    """

    def __init__(self, *, weight, z_pins, device, params):
        self.z_pins = z_pins
        self.device = device
        self.mse_loss = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(params(), lr=weight, weight_decay=1e-8)

    def batch(self, x_pt, y_pt, model, both):
        # This value is a multiplier of the hypersphere Z radio to use in sampling. It should be
        # a number slightly larger than 1.0.
        ZPIN_SAMPLE_RADIO = 1.2

        z_pt = model.forward_y(x_pt, y_pt)
        y_pt_res = model.forward_z(x_pt, z_pt)

        self.optimizer.zero_grad()
        lossy = self.mse_loss(y_pt_res, y_pt)
        if both:
            z_pins = self.z_pins.sample(x_pt.shape[0], ZPIN_SAMPLE_RADIO)
            z_pins_y = model.forward_z(x_pt, z_pins)
            z_pins_ret = model.forward_y(x_pt, z_pins_y)

            lossz = self.mse_loss(z_pins_ret, z_pins)
        else:
            lossz = 0.0

        loss = lossy + lossz
        loss.backward()
        self.optimizer.step()

        return loss, z_pt


class Trainer:
    """
    This class implements PMT training.
    """

    def __init__(
        self,
        *,
        trainer_config,
        z_pins,
        params,
        invparams,
        writer,
        device,
    ):
        self.z_pins = z_pins
        self.scalar_calculator = MovementScalarCalculator(z_pins=z_pins, device=device)

        self.movement = trainer_config["movement"]
        self.max_weight = trainer_config.get("max_weight")
        self.gamma_y = trainer_config["gamma_y"]
        self.gamma_p = trainer_config.get("gamma_p", 1.0)
        self.gamma_z = trainer_config.get("gamma_z", 1.0)

        self.pretrain_epochs = trainer_config.get("pretrain_epochs", 30)
        self.pretrain_recenter_z_epochs = trainer_config.get(
            "pretrain_recenter_z_epochs", 10
        )

        self.device = device

        self.learning_rate = trainer_config["learning_rate"]
        self.step_size = trainer_config["step_size"]
        self.num_z_pins = trainer_config.get("num_z_pins", 1)

        # Create an adam optimizer.
        self.optimizer = optim.Adam(
            [{"params": params()}, {"params": invparams()}],
            lr=self.learning_rate,
            weight_decay=1e-6,
        )

        # Create a scheduler to decrease the learning rate.
        self.scheduler_movement = trainer_config["scheduler_movement"]
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.step_size,
            gamma=trainer_config["scheduler_weight"],
        )
        self.writer = writer

        # Create an Weighted Mean Squared Error (WMSE) loss function for the targets.
        def weighted_mae_loss(inputs, targets, weights):
            # MAE seems to work best with PM training since bigger differences
            # should be treated linearly and differences are never 0.
            # MAE
            sqerr = torch.abs(inputs.view(inputs.shape[0], -1) - targets)

            out = sqerr * weights
            loss = out.sum(dim=1).mean()

            return loss

        def mse_loss(inputs, targets):
            # MSE
            sqerr = (inputs - targets) ** 2

            loss = sqerr.sum(dim=1).mean()

            return loss

        self.loss_mae = weighted_mae_loss
        self.loss_mse = mse_loss

    def pretrain(self, model, datasets):
        WEIGHT = 1e-3
        invtrainer = PreTrainer(
            params=model.parameters,
            weight=WEIGHT,
            z_pins=self.z_pins,
            device=self.device,
        )

        recon_and_invrecon = False
        for epoch in range(self.pretrain_epochs):
            for x, y in datasets.data_loader_train:
                if epoch and epoch % self.pretrain_recenter_z_epochs == 0:
                    offset = last_z_pt.mean(dim=0)
                    radio = ((last_z_pt - offset) ** 2).sum(dim=1).sqrt().mean().item()

                    self.z_pins.set_radio_and_offset(radio, offset.detach())
                    print(f"Recentered Z {offset=} {radio=}")
                    recon_and_invrecon = True
                loss, last_z_pt = invtrainer.batch(x, y, model, recon_and_invrecon)
                print(f"Pretrain {loss=} {epoch=}")

    def step(self, epoch):
        """
        This method is called once for every epoch.
        """
        # pylint: disable=unused-argument
        self.scheduler.step()
        if epoch > 0 and epoch % self.step_size == 0:
            self.writer.save()
            self.movement *= self.scheduler_movement

    def calculate_scalars(self, z_pins_line, t_sizes, actual_behind_index):
        zeros = torch.zeros(z_pins_line.shape, device=self.device)

        a = t_sizes - z_pins_line
        a = a.clamp(max=t_sizes, min=zeros)

        b = z_pins_line
        b = b.clamp(max=t_sizes, min=zeros)

        # Assert that a + b = t_sizes.
        isnan = z_pins_line.isnan()
        assert (a[~isnan] + b[~isnan] - t_sizes[~isnan] < EPSILON).all()

        scalars = t_sizes / (2.0 * a)
        scalars[actual_behind_index] = t_sizes[actual_behind_index] / (
            2.0 * b[actual_behind_index]
        )

        # This index is true on zpins where reversing the vector results in
        # a smaller distance to zspace surface.
        smaller_flip_index = a > b

        max_w_bp = t_sizes / (2.0 * a)
        # Get a clone before the flip to assert that the flip makes it larger.
        # max_w_bp_clone = max_w_bp.clone()
        max_w_bp[smaller_flip_index] = t_sizes[smaller_flip_index] / (
            2.0 * b[smaller_flip_index]
        )
        # if max_w_bp_clone.mean() >= max_w_bp.mean():
        #    assert False

        # Adjust the weights.
        #####################################################################
        adjustment = self.max_weight / max_w_bp
        adjustment = adjustment.clamp(max=1.0)
        scalars *= adjustment

        # If the max weight is inf (edge zpin), then the adjustment is 0.0, if
        # scalars is nan then it was inf before multiplying by adjustment and means
        # it is moved beyond the edge, so adjust to max_weight.
        # TODO: check logic maybe use actual_behind_index instead of scalars.isnan
        scalars[max_w_bp.isinf() & scalars.isnan()] = self.max_weight
        #####################################################################

        assert not scalars[~isnan].isinf().any()

        return scalars

    def backprop(self, x_bp, z_pins_bp, y_bp, w_bp, model, x_pt, y_pt, iteration):
        """
        This method runs backpropagation on the neural network.
        """
        model.train()

        # Zero out all the gradients before running the backward pass.
        model.zero_grad()

        ############################ loss_z #######################################
        # It is necessary to sample from the entire radius plus the movement to ensure that
        # the targets land in places in Z that are inverse reconstruciotn trained.
        z_pins = self.z_pins.sample(x_pt.shape[0], 1.0 + self.movement)
        z_pins_y = model.forward_z(x_pt, z_pins)
        z_pins_z = model.forward_y(x_pt, z_pins_y)
        lossz = self.loss_mse(z_pins_z, z_pins)
        ###########################################################################

        ############################ loss_y #######################################
        y_pt_z = model.forward_y(x_pt, y_pt)
        y_pt_y = model.forward_z(x_pt, y_pt_z)
        lossy = self.loss_mse(y_pt_y, y_pt)
        ###########################################################################

        ############################ loss_p #######################################
        # Run the forward pass.
        y_predict = model.forward_z(x_bp, z_pins_bp)

        # Compute the loss.
        loss_p = self.loss_mae(y_predict, y_bp, w_bp)
        ###########################################################################

        loss = loss_p * self.gamma_p + lossy * self.gamma_y + lossz * self.gamma_z

        # Run the backward pass.
        loss.backward()

        # Run the optimizer.
        self.optimizer.step()

        # Log the losses.
        self.writer.log_loss_z(lossz / (z_pins.shape[0] * z_pins_y.shape[1]), iteration)
        # Loss-y has to be recalculated using the mean. As opposed to sum -> mean.
        lossy = ((y_pt - y_pt_y) ** 2).mean()
        self.writer.log_loss_y(lossy, iteration)
        self.writer.log_loss_p(loss_p, iteration)

    def batch(self, x_pt, y_pt, model, iteration):
        z_pins_offset = self.z_pins.z_offset
        z_pins_radio = self.z_pins.z_radio

        num_data_points = y_pt.shape[0]
        y_dimensions = y_pt.shape[1]
        z_dimensions = self.z_pins.z_dimensions

        # Set the model to "eval" while we calculate our z-targets.
        model.eval()

        # Get the z -> match and calculate which points are inside Z hypersphere.
        with torch.no_grad():
            z_y_match = model.forward_y(x_pt, y_pt)
        z_y_match_radios = ((z_y_match - z_pins_offset) ** 2).sum(dim=1).sqrt()
        outside = z_y_match_radios > z_pins_radio
        num_outside = torch.where(outside)[0].shape[0]

        self.writer.log_out_ratio(num_outside / num_data_points, iteration)

        # Sample z-dirs.
        z_pins_angles = (
            self.z_pins.sample(num_data_points * self.num_z_pins) - z_pins_offset
        )
        z_pins_angles = z_pins_angles.view(
            (self.num_z_pins, num_data_points, z_dimensions)
        )
        D_o, _ = get_unit_vector_and_magnitude(z_pins_angles)

        # Using the z-dirs, sample z-pins from the z-lines.
        z_pins_line = torch.rand((self.num_z_pins, num_data_points), device=self.device)

        z_pins_line = z_pins_line * 1.1 - 0.05
        z_pins_line = z_pins_line.clamp(min=0.0, max=1.0)

        z_y_match_x = z_y_match

        t_pos_actual, t_neg_actual, nidx = self.scalar_calculator.get_distances(
            D_o, z_y_match_x.unsqueeze(0).repeat((self.num_z_pins, 1, 1))
        )

        t_neg_actual = -t_neg_actual
        t_sizes = t_pos_actual + t_neg_actual
        z_pins_line *= t_sizes

        z_pins = z_y_match_x + D_o * (z_pins_line - t_neg_actual).unsqueeze(2)

        # Switch the z-dirs to point in the direction of the selected z-pins.
        behind_index = z_pins_line > t_neg_actual
        D_o[behind_index] = -D_o[behind_index]

        # Calculate the z-targets.
        z_pins_target = z_pins + D_o * self.movement * z_pins_radio

        y_bpx = get_z_pin_preds_specific(model=model, x_pt=x_pt, z_pins=z_pins_target)
        y_bp = y_bpx.reshape((y_bpx.shape[0] * y_bpx.shape[1], y_bpx.shape[2]))

        # Calculate the scalars.
        scalars = self.calculate_scalars(z_pins_line, t_sizes, behind_index)

        # This matrix will have the weights to be used on the loss function.
        # dimensions: (z-pins, data points, output dimensions)
        w_bp = scalars.view((scalars.shape[0] * scalars.shape[1], 1))

        # dimensions: (z-pins * data points, input dimensions)
        repeat_shape = [1 for _ in range(len(x_pt.shape))]
        repeat_shape[0] = z_pins.shape[0]
        x_bp = x_pt.repeat(tuple(repeat_shape))

        nidx = nidx.view((nidx.shape[0] * nidx.shape[1]))
        x_bp_m = x_bp[nidx]
        z_pins_bp_m = z_pins.view((self.num_z_pins * num_data_points, z_dimensions))[
            nidx
        ]
        y_bp_m = y_bp[nidx]
        w_bp_m = w_bp[nidx]
        self.writer.summwriter.add_scalar("scalarsmean", w_bp_m.mean(), iteration)
        self.writer.summwriter.add_scalar("scalarsmax", w_bp_m.max(), iteration)

        self.backprop(
            x_bp=x_bp_m,
            z_pins_bp=z_pins_bp_m,
            y_bp=y_bp_m,
            w_bp=w_bp_m,
            model=model,
            x_pt=x_pt,
            y_pt=y_pt,
            iteration=iteration,
        )
