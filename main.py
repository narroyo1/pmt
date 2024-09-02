import time

import torch

import experiments

from datasets import DataSets
from plotter import Plotter, SampleRender
from model import Model, LayersBuilder
from zpins import ZPins
from trainer import Trainer
from vae_trainer import VAETrainer, VAEZPins, VariationalAutoencoder

from tester import Tester
from writer import Writer

EPOCHS_BETWEEN_SAVES = 1_000


def build_model(method, model_config):
    # Create the writer, this object will write information that can be browsed using tensorboard.
    writer = Writer()

    if "layers_builder" in model_config:
        layers_builder = model_config["layers_builder"]
    else:
        hidden_layers = model_config["hidden_layers"]
        hidden_size = model_config["hidden_size"]
        function = model_config["function"]
        layers_builder = LayersBuilder(
            hidden_layers=hidden_layers, hidden_size=hidden_size, function=function
        )

    if method == "pmt":
        z_pins = ZPins(z_pins_config=experiment["z_pins_config"], device=device)

        model = Model(
            z_space_size=z_pins.z_dimensions,
            x_space_size=datasets.x_dimensions,
            y_space_size=datasets.y_dimensions,
            device=device,
            layers_builder=layers_builder,
        ).to(device=device)

        trainer = Trainer(
            trainer_config=experiment["trainer_config"],
            z_pins=z_pins,
            params=model.net.parameters,
            invparams=model.invnet.parameters,
            writer=writer,
            device=device,
        )
    elif method == "vae":
        z_pins = VAEZPins(z_pins_config=experiment["z_pins_config"], device=device)
        model = VariationalAutoencoder(
            z_space_size=z_pins.z_pins_dimensions,
            x_space_size=datasets.x_dimensions,
            y_space_size=datasets.y_dimensions,
            device=device,
            layers_builder=layers_builder,
        )
        trainer = VAETrainer(
            trainer_config=experiment["trainer_config"],
            z_pins=z_pins,
            writer=writer,
            device=device,
            params=model.parameters,
        )

    # Create the plotter, this object will render all plots to the filesystem.

    if experiment.get("plotter_config", {}).get("sample_render", False):
        plotter = SampleRender(datasets=datasets, device=device)
    else:
        plotter = Plotter(datasets=datasets)

    return model, trainer, plotter, writer, z_pins


# %%

# Select the desired experiment.
experiment = experiments.EXPERIMENT_SQUARES
# Select the desired training method. It may be "pmt" or "vae"
method = "pmt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_config = experiment["model"]
dataset_config = experiment["dataset_config"]

datasets = DataSets.build_datasets(dataset_config=dataset_config, device=device)

model, trainer, plotter, writer, z_pins = build_model(method, model_config)

# Create the tester, this object will run emd tests.
tester = Tester(
    experiment=experiment["tester_config"],
    z_pins=z_pins,
    x_test=datasets.x_test,
    y_test=datasets.y_test,
    y_train=datasets.y_train,
    plotter=plotter,
    writer=writer,
    device=device,
)


# %%
def save_checkpoint(epoch):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "epoch": epoch,
            "z_pins_radio": z_pins.z_radio,
            "z_pins_offset": z_pins.z_offset,
            "movement": trainer.movement,
            "gamma_y": trainer.gamma_y,
            "gamma_p": trainer.gamma_p,
            "gamma_z": trainer.gamma_z,
        },
        f"checkpoint{epoch}.pt",
    )


# %%
epoch = 0

plotter.plot_test_data()
trainer.pretrain(model, datasets)

# %%

iteration = 0

# Iterate running the training algorithm.
for epoch in range(epoch, experiment.get("num_epochs", 10_000)):
    start = time.time()
    for x, y in datasets.data_loader_train:
        trainer.batch(x_pt=x, y_pt=y, model=model, iteration=iteration)
        iteration += 1

    trainer.step(epoch=epoch)

    model.eval()
    tester.step(model=model, epoch=epoch)

    end = time.time()
    print(f"epoch: {epoch} elapsed time: {end - start}")

    # Save the state in case we wan't to resume from certain point.
    if epoch % EPOCHS_BETWEEN_SAVES == 0:
        save_checkpoint(epoch)


# %%

filename = "checkpoint3511.pt"
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint["model"])
trainer.optimizer.load_state_dict(checkpoint["optimizer"])
epoch = checkpoint["epoch"] + 1
model.train()

trainer.movement = checkpoint["movement"]
trainer.gamma_y = checkpoint["gamma_y"]
trainer.gamma_p = checkpoint["gamma_p"]
trainer.gamma_z = checkpoint["gamma_z"]
z_pins_radio = checkpoint["z_pins_radio"]
z_pins_offset = checkpoint["z_pins_offset"]
z_pins.set_radio_and_offset(z_pins_radio, z_pins_offset)


# %%
