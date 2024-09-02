"""
This module contains class DataSets.
"""
# pylint: disable=bad-continuation

import gzip
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import sample_random, to_tensor

BATCH_SIZE = 3072


class FunctionDataSet(Dataset):
    """
    This class implements a dataset by wrapping x and y data arrays.
    """

    def __init__(self, x_np, y_np, device):
        self.length = x_np.shape[0]

        # Convert numpy arrays to pytorch tensors.
        self.x_data = to_tensor(x_np, device)
        self.y_data = to_tensor(y_np, device)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


class DataSets:
    """
    This class creates train and test datasets from the provided dimensions and functions.
    """

    def __init__(
        self,
        *,
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size,
        device,
    ):
        """
        This function initializes the training and testing data sets.
        """
        self.x_test = x_test
        self.y_test = y_test
        self.x_dimensions = x_test.shape[1]
        self.y_dimensions = y_test.shape[1]

        self.y_train = y_train

        dataset_train = FunctionDataSet(x_np=x_train, y_np=y_train, device=device)

        self.data_loader_train = DataLoader(
            dataset=dataset_train, batch_size=batch_size, shuffle=True
        )

    @staticmethod
    def build_datasets(dataset_config, device):
        # If using a custom function to create the dataset.
        if "function" in dataset_config:
            datasets = DataSets._function(
                dataset_config["function"], batch_size=BATCH_SIZE, device=device
            )
        # If using a generated dataset.
        elif "filename" not in dataset_config:
            datasets = DataSets._generated_dataset(
                dataset_config=dataset_config,
                batch_size=BATCH_SIZE,
                device=device,
            )
        # If using a real data dataset.
        elif dataset_config["filename"].endswith("csv"):
            datasets = DataSets._load_csv(
                dataset_config=dataset_config, batch_size=BATCH_SIZE, device=device
            )
        elif dataset_config["filename"].endswith("gz"):
            datasets = DataSets._load_gz(
                dataset_config=dataset_config, batch_size=BATCH_SIZE, device=device
            )

        return datasets

    @staticmethod
    def _function(function, batch_size, device):
        TRAIN_SIZE = 31013
        TEST_SIZE = 9007

        y_train = function(n_samples=TRAIN_SIZE)[0]
        y_test = function(n_samples=TEST_SIZE)[0]

        return DataSets(
            x_train=np.empty([y_train.shape[0], 0]),
            y_train=y_train,
            x_test=np.empty([y_test.shape[0], 0]),
            y_test=y_test,
            batch_size=batch_size,
            device=device,
        )

    @staticmethod
    def _generated_dataset(
        dataset_config,
        batch_size,
        device,
    ):
        """
        This named constructor builds a DataSet from a pair of base and noise functions.
        """
        # Use coprime numbers to prevent any matching points between train and test.
        TRAIN_SIZE = 31013
        TEST_SIZE = 9007
        # TEST_SIZE = 6007

        base_function = dataset_config["base_function"]
        noise_function = dataset_config["noise_function"]
        x_range_train = dataset_config["x_range_train"]
        x_range_test = dataset_config["x_range_test"]

        # target_function_desc = "{}/{}".format(base_function.name, noise_function.name)
        target_function = lambda x, y: noise_function(*base_function(x, y))

        def create_dataset(function, size, ranges):
            """
            This function takes a composed function a size and a range and
            then creates an artificial dataset based on the function.
            """
            x_np = sample_random(ranges, size)
            y_np = np.zeros((x_np.shape[0], base_function.y_space_size))
            x_np, y_np = function(x_np, y_np)

            return x_np, y_np

        # Create the training dataset.
        x_train, y_train = create_dataset(target_function, TRAIN_SIZE, x_range_train)

        # Create the test dataset.
        x_test, y_test = create_dataset(target_function, TEST_SIZE, x_range_test)

        return DataSets(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            batch_size=batch_size,
            device=device,
        )

    @staticmethod
    def _load_gz(dataset_config, batch_size, device):
        """
        Extract the images into a 4D tensor [image index, y, x, channels].
        Values are rescaled from [0, 255] down to [-0.5, 0.5].
        """
        filename = dataset_config["filename"]
        IMAGE_SIZE = 28
        NUM_CHANNELS = 1
        PIXEL_DEPTH = 255
        num_images = 60000
        norm_shift = False
        norm_scale = True
        TEST_SIZE = 0.2

        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            if norm_shift:
                data = data - (PIXEL_DEPTH / 2.0)
            if norm_scale:
                data = data / PIXEL_DEPTH
            data = data.reshape(num_images, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            data = np.reshape(data, [num_images, -1])
        data = data[:40000]

        train, test = train_test_split(data, test_size=TEST_SIZE)

        return DataSets(
            x_train=np.empty([train.shape[0], 0]),
            y_train=train,
            x_test=np.empty([test.shape[0], 0]),
            y_test=test,
            batch_size=batch_size,
            device=device,
        )

    @staticmethod
    def _load_csv(dataset_config, batch_size, device):
        """
        This named constructor builds a Dataset from a time series.
        """

        input_columns = dataset_config["input_columns"]
        output_columns = dataset_config["output_columns"]
        filename = dataset_config["filename"]

        all_columns = input_columns + output_columns

        def create_dataset(data_file, combined_size, test_size):
            pandas_dataframe = pd.read_csv(data_file)

            pandas_dataframe = pandas_dataframe.dropna(how="any", subset=all_columns)

            for transformation_tuple in dataset_config.get("transformations", []):
                if isinstance(transformation_tuple, tuple):
                    transformation = lambda pdf, outcols: transformation_tuple[0](
                        pdf, outcols, *transformation_tuple[1:]
                    )
                else:
                    transformation = transformation_tuple
                pandas_dataframe = transformation(pandas_dataframe, output_columns)

            if len(pandas_dataframe) > combined_size:
                pandas_dataframe = pandas_dataframe.sample(combined_size)
            # df_size = len(pandas_dataframe)
            train, test = train_test_split(pandas_dataframe, test_size=test_size)

            x_train = train.loc[:, input_columns].values
            y_train = train.loc[:, output_columns].values

            x_test = test.loc[:, input_columns].values
            y_test = test.loc[:, output_columns].values

            scaler_x = StandardScaler().fit(x_train) if x_train.shape[1] else None
            scaler_y = StandardScaler().fit(y_train)
            if scaler_x:
                x_train = scaler_x.transform(x_train)
                x_test = scaler_x.transform(x_test)

            y_train = scaler_y.transform(y_train)
            y_test = scaler_y.transform(y_test)

            y_test *= 10.0
            y_train *= 10.0

            return x_train, y_train, x_test, y_test

        COMBINED_SIZE = 47013
        TEST_SIZE = 0.2
        x_train, y_train, x_test, y_test = create_dataset(
            filename, COMBINED_SIZE, TEST_SIZE
        )

        return DataSets(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            batch_size=batch_size,
            device=device,
        )
