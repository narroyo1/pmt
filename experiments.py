"""
This module has a set of preset experiments to test the model.
"""
import numpy as np

import functions as func
from sklearn import datasets


def trim_transformation(df, columns, loquant, hiquant):
    """
    Trims a dataset in all dimensions.
    """
    assert 0.0 <= loquant <= 1.0 and 0.0 <= hiquant <= 1.0
    index = None
    for column in columns:
        q_hi = df[column].quantile(1.0 - hiquant)
        q_lo = df[column].quantile(0.0 + loquant)
        new_index = (df[column] < q_hi) & (df[column] > q_lo)
        index = new_index if index is None else index & new_index

    df = df[index]

    return df


def fillin_transformation(df, columns, scale):
    """
    Fills in datasets with integer values or small number of decimals.
    """
    assert scale > 0.0
    for column in columns:
        df[column] += np.random.uniform(0.0, scale, len(df))
    return df


def scale_transformation(df, columns, scale):
    """
    Scales in the daataset in all dimensions.
    """
    for column in columns:
        df[column] *= scale
    return df


EXPERIMENT_BLOBS = {
    "tester_config": {"skip_epochs": 25},
    "num_epochs": 15000,
    "trainer_config": {
        # Scheduler parameters:
        "scheduler_weight": 0.97,
        "scheduler_movement": 0.9,
        "step_size": 800,
        # Model parameters:
        "max_weight": 10.0,
        "movement": 0.04,
        "gamma_y": 1.5,
        "gamma_p": 3.6,
        "gamma_z": 1.0,
        "learning_rate": 1e-4,
        "kld_weight": 0.02,
    },
    "z_pins_config": {"z_pins_dimensions": 2, "z_pins_radio": 8.0},
    "dataset_config": {
        "function": lambda n_samples: datasets.make_blobs(
            n_samples=n_samples,
            centers=[[1.3, 2.0], [-1.2, -1.4], [3.1, -1.9], [1.2, -0.3], [3.9, -0.2]],
            cluster_std=[0.4, 0.3, 0.8, 0.5, 0.3],
            random_state=0,
        )
    },
    "model": {"hidden_size": 128, "hidden_layers": 4, "function": "lrelu"},
}

EXPERIMENT_SQUARES = {
    "tester_config": {"skip_epochs": 25},
    "num_epochs": 3500,
    "trainer_config": {
        "scheduler_weight": 0.95,
        "scheduler_movement": 0.8,
        "step_size": 200,
        "movement": 0.04,
        "max_weight": 6.0,
        "gamma_y": 1.0,
        "gamma_p": 5.0,
        "gamma_z": 1.0,
        "learning_rate": 1e-4 / 2.0,
        "num_z_pins": 1,
        "kld_weight": 1.0,
    },
    "z_pins_config": {"z_pins_dimensions": 2, "z_pins_radio": 8.0},
    "dataset_config": {
        "base_function": func.binder(
            func.fn_rectangle, side1=15.0, side2=15.0, y_space_size=2
        ),
        "noise_function": func.binder(func.fn_rescale),
        "x_range_train": np.array([[-0.001, 0.001]]),
        "x_range_test": np.array([[-0.0008, 0.0008]]),
    },
    "model": {"hidden_size": 256, "hidden_layers": 4, "function": "lrelu"},
}


class MNISTLayerBuilder:
    def __init__(self):
        self.hidden_size = 784

    def build_invnetwork(self, *, input_size, x_space_size, output_size, device):
        import torch.nn as nn

        keep_prob = 0.99
        return [
            nn.Linear(input_size, self.hidden_size),
            nn.ELU(inplace=True),
            nn.Dropout(1 - keep_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(inplace=True),
            nn.Dropout(1 - keep_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(inplace=True),
            nn.Dropout(1 - keep_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(1 - keep_prob),
            nn.Linear(self.hidden_size, output_size),
        ]

    def build_network(self, *, input_size, x_space_size, output_size, device):
        import torch.nn as nn

        keep_prob = 0.99
        return [
            nn.Linear(input_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(1 - keep_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(),
            nn.Dropout(1 - keep_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(),
            nn.Dropout(1 - keep_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(),
            nn.Dropout(1 - keep_prob),
            nn.Linear(self.hidden_size, output_size),
            nn.Sigmoid(),
        ]


EXPERIMENT_MNIST = {
    "num_epochs": 8000,
    "dataset_config": {
        "filename": "mnist/train-images-idx3-ubyte.gz",
        "transformations": [],
    },
    "z_pins_config": {
        "z_pins_dimensions": 4,
        "z_pins_radio": 8.0,
    },
    "trainer_config": {
        "scheduler_weight": 0.95,
        "scheduler_movement": 0.9,
        "max_weight": 30.0,
        "movement": 0.04,
        "learning_rate": 1e-4,
        "step_size": 700,
        "gamma_y": 4.0,
        "gamma_p": 1.8,
        "gamma_z": 0.5,
        "num_z_pins": 1,
    },
    "plotter_config": {"sample_render": True},
    "tester_config": {"skip_epochs": 25, "emd_test": False},
    "model": {"layers_builder": MNISTLayerBuilder()},
}

EXPERIMENT_HUMAN_BEHAVIOR = {
    "tester_config": {"skip_epochs": 25},
    "num_epochs": 15000,
    "trainer_config": {
        "scheduler_weight": 0.95,
        "scheduler_movement": 0.9,
        "step_size": 600,
        "max_weight": 8.0,
        "movement": 0.04,
        "gamma_y": 1.5,
        "gamma_p": 2.5,
        "gamma_z": 1.0,
        "learning_rate": 1e-4 / 2.0,
        "gamma": 0.9999999999,
        "num_z_pins": 1,
        "kld_weight": 2.0,
    },
    "z_pins_config": {
        "z_pins_dimensions": 3,
        "z_pins_radio": 8.0,
    },
    # datasets
    "dataset_config": {
        "filename": "datasets/mhealth_raw_data.csv",
        "input_columns": [],
        "output_columns": ["alx", "aly", "alz"],
        "transformations": [(trim_transformation, 0.1, 0.1)],
    },
    "model": {"hidden_size": 512, "hidden_layers": 5, "function": "lrelu"},
}
