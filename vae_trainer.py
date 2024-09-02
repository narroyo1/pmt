import torch

from torch import nn
from torch import optim
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(
        self, z_space_size, x_space_size, y_space_size, device, layers_builder
    ):
        super(Encoder, self).__init__()
        layers = layers_builder.build_network(
            input_size=y_space_size,
            x_space_size=x_space_size,
            output_size=z_space_size * 2,
            device=device,
        )
        self.net = nn.Sequential(*layers).to(device)
        self.z_space_size = z_space_size

    def forward(self, x_pt, y_pt):
        # If there is data on x merge it with y as an input.
        if x_pt.shape[1] > 0:
            mixed_pt = torch.cat(
                (x_pt.view(x_pt.size(0), -1), y_pt.view(y_pt.size(0), -1)), dim=1
            )
        else:
            mixed_pt = y_pt
        h_ = self.net(mixed_pt)
        mean = h_[:, : self.z_space_size]
        log_var = h_[:, self.z_space_size :]

        return mean, log_var


class Decoder(nn.Module):
    def __init__(
        self, z_space_size, x_space_size, y_space_size, device, layers_builder
    ):
        super(Decoder, self).__init__()
        layers = layers_builder.build_network(
            input_size=z_space_size,
            x_space_size=x_space_size,
            output_size=y_space_size,
            device=device,
        )
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, x_pt, y_pt):
        # If there is data on x merge it with y as ain input
        if x_pt.shape[1] > 0:
            mixed_pt = torch.cat(
                (x_pt.view(x_pt.size(0), -1), y_pt.view(y_pt.size(0), -1)), dim=1
            )
        else:
            mixed_pt = y_pt

        return self.net(mixed_pt)


class VariationalAutoencoder(nn.Module):
    def __init__(
        self, *, z_space_size, x_space_size, y_space_size, device, layers_builder
    ):
        super(VariationalAutoencoder, self).__init__()
        self.device = device
        self.encoder = Encoder(
            z_space_size, x_space_size, y_space_size, device, layers_builder
        ).to(device)
        self.decoder = Decoder(
            z_space_size, x_space_size, y_space_size, device, layers_builder
        ).to(device)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z_pt = mean + var * epsilon

        return z_pt

    def forward(self, x_pt, y_pt):
        mean, log_var = self.encoder(x_pt, y_pt)
        z_pt = self.reparameterization(mean, torch.exp(0.5 * log_var))

        return self.decoder(x_pt, z_pt), mean, log_var

    def reconstruct(self, x_pt, y_pt):
        with torch.no_grad():
            mean, log_var = self.encoder(x_pt, y_pt)
            z_pt = mean
            return self.decoder(x_pt, z_pt)

    def forward_z(self, x_pt, z_pt):
        return self.decoder(x_pt, z_pt)

    def forward_y(self, x_pt, y_pt):
        return self.encoder(x_pt, y_pt)


class VAETrainer:
    def __init__(self, *, trainer_config, z_pins, writer, device, params):
        self.optimizer = torch.optim.Adam(params(), lr=0.0001)
        self.writer = writer
        self.kld_weight = trainer_config.get("kld_weight", 1.0)
        self.step_size = trainer_config["step_size"]
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.step_size,
            gamma=0.9,
        )

    def pretrain(self, model, datasets):
        pass

    def batch(self, x_pt, y_pt, model, iteration):
        model.train()
        self.optimizer.zero_grad()
        y_hat, mean, log_var = model.forward(x_pt, y_pt)

        loss_mse = ((y_pt - y_hat) ** 2).sum()

        loss_kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        loss = loss_mse + self.kld_weight * loss_kld
        loss.backward()
        self.optimizer.step()

        # Write the reconstruction loss.
        y_hat = model.reconstruct(x_pt, y_pt)
        loss_mse = ((y_pt - y_hat) ** 2).mean()
        self.writer.log_loss_y(loss_mse, iteration)

    def step(self, epoch):
        self.scheduler.step()
        if epoch > 0 and epoch % self.step_size == 0:
            self.writer.save()


class VAEZPins:
    def __init__(self, *, z_pins_config, device):
        self.z_pins_dimensions = z_pins_config["z_pins_dimensions"]
        self.device = device

    def sample(self, size):
        z = torch.randn(size, self.z_pins_dimensions)

        z = z.to(self.device)
        return z
