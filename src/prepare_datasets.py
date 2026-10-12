import os
import pickle

import numpy as np

from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split


DATASETS = [
    "FordA",
    "FordB",
    "Adiac",
    "Libras",
    "RacketSports",
]

OUTPUT_PATH = "./data"


def convert_shape(x):
    """
    aeon:
        (samples, channels, timesteps)

    our pipeline:
        (samples, timesteps, features)
    """

    return np.transpose(x, (0, 2, 1))


def standardize(train, val, test):
    """
    Standardization using TRAIN statistics only.
    """

    mean = np.mean(
        train,
        axis=(0, 1),
        keepdims=True,
    )

    std = np.std(
        train,
        axis=(0, 1),
        keepdims=True,
    )

    std[std == 0] = 1.0

    train = (train - mean) / std
    val = (val - mean) / std
    test = (test - mean) / std

    return train, val, test


def prepare_dataset(dataset_name):

    print(f"\n--- {dataset_name} ---")

    # Load complete dataset
    x, y = load_classification(
        name=dataset_name,
    )

    x = convert_shape(x)

    print("Full dataset shape:", x.shape)
    print("Labels shape:", y.shape)

    # --------------------------------------------------------
    # First split:
    # about 66% train_all / 33% test
    # --------------------------------------------------------

    x_train_all, x_test, y_train_all, y_test = train_test_split(
        x,
        y,
        test_size=0.33,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------------
    # Second split:
    # train_all -> about 66% train / 33% validation
    # --------------------------------------------------------

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all,
        y_train_all,
        test_size=0.33,
        random_state=42,
        stratify=y_train_all,
    )

    # --------------------------------------------------------
    # Standardization using only TRAIN statistics
    # --------------------------------------------------------

    x_train, x_val, x_test = standardize(
        x_train,
        x_val,
        x_test,
    )

    # Important:
    # x_train_all must be standardized using the same
    # statistics used for x_train.
    #
    # Reconstruct it from train + validation.
    x_train_all = np.concatenate(
        [x_train, x_val],
        axis=0,
    )

    y_train_all = np.concatenate(
        [y_train, y_val],
        axis=0,
    )

    print("Train shape:", x_train.shape)
    print("Validation shape:", x_val.shape)
    print("Train all shape:", x_train_all.shape)
    print("Test shape:", x_test.shape)

    dataset = (
        x_train_all,
        y_train_all,
        x_test,
        y_test,
        x_train,
        x_val,
        y_train,
        y_val,
    )

    filename = os.path.join(
        OUTPUT_PATH,
        f"{dataset_name}_dataset.p",
    )

    with open(filename, "wb") as f:
        pickle.dump(dataset, f)

    print("Saved:", filename)


if __name__ == "__main__":

    os.makedirs(
        OUTPUT_PATH,
        exist_ok=True,
    )

    for dataset_name in DATASETS:
        prepare_dataset(dataset_name)