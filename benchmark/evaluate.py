""" Module contains functions for evaluating models """

import argparse
import os
import warnings
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

""" Constants """

# Default paths
DEFAULT_MODEL_PATH = "./models/bce_masks_split"
DEFAULT_DATA_PATH = "./benchmark/masks_split/test"
DEFAULT_ML_100K_PATH = "/data/raw/ml-100k/"

# Data-specific constants
NUM_MOVIES = 1682
BASIC_USER_FEATURES = 3
TOTAL_USER_FEATURES = BASIC_USER_FEATURES + 19


# Model constants
INPUT_SIZE = TOTAL_USER_FEATURES + NUM_MOVIES
DEVICE = torch.device("cpu")


class Logger:
    """Manage log messages"""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    def log(self, message: str):
        """Log message to console

        Args:
            message (str): message to log
        """
        if self.verbose:
            print(message)


class RecommendationDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, verbose: bool):
        self.df = df.drop(columns=["user_id"])
        features = []
        inputs = []
        targets = []

        loop = df.iterrows()
        if verbose:
            loop = tqdm(loop, total=len(df))

        for _, row in loop:
            features.append(
                row[:BASIC_USER_FEATURES].tolist() + literal_eval(row["genres"])
            )
            inputs.append(literal_eval(row["input"]))
            targets.append(literal_eval(row["output"]))

        self.features = np.array(features)

        # normalize ratings
        self.inputs = np.array(inputs) / 5
        self.targets = np.array(targets) / 5

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_ratings = self.inputs[idx]
        input_data = np.concatenate([self.features[idx], input_ratings])
        mask = input_ratings == 0
        return input_data, mask, self.targets[idx]

    def __len__(self) -> int:
        return len(self.df)


class RecSys(nn.Module):
    def __init__(
        self,
        hidden_dim1: int = 1024,
        hidden_dim2: int = 1024,
        hidden_dim3: int = 1024,
    ):
        super(RecSys, self).__init__()

        self.d1 = nn.Dropout(0.1)
        self.d2 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(INPUT_SIZE, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, NUM_MOVIES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.relu(self.fc3(x))

        return F.sigmoid(self.fc4(x))


""" Save & load functions """


def load_model(model_path: str, logger: Logger) -> nn.Module:
    logger.log(f"Loading model from '{model_path}'...")
    model = torch.load(model_path)
    model.eval()
    logger.log("Success!")
    return model


def load_and_build_dataset(
    data_path: str, file_mode: bool, logger: Logger
) -> RecommendationDataset:
    if file_mode:
        logger.log(f"File mode is ON. Reading '{data_path}' as file...")
        df = pd.read_csv(data_path)
    else:
        logger.log(f"File mode is OF. Reading from folder '{data_path}'...")
        df = pd.concat(
            [
                pd.read_csv(os.path.join(data_path, file_name))
                for file_name in os.listdir(data_path)
            ]
        )
    logger.log("Success!")

    logger.log("Building dataset...")
    dataset = RecommendationDataset(df, logger.verbose)
    logger.log("Success!")
    return dataset


""" Model functions """


def get_unseen_on_input_data(
    input_rating: np.ndarray, movie_ratings: np.ndarray
) -> np.ndarray:
    unseen_ratings = movie_ratings.copy()
    seen_indices = np.nonzero(input_rating > 0)[0]
    unseen_ratings[seen_indices] = 0
    return unseen_ratings


def get_single_output(
    model: nn.Module,
    input_data: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        model.eval()
        input_tensor = torch.Tensor([input_data]).to(DEVICE)
        model_out = model(input_tensor)

    return model_out[0].cpu().numpy()


""" Main evaluation function """


def evaluate():
    """Evaluate  model"""
    global DEVICE

    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluates given model")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        dest="model",
        default=DEFAULT_MODEL_PATH,
        help="path to pytorch model)",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        dest="save_path",
        default="./benchmark/data)",
        help="relative path to save generated results (default: ./benchmark/data)",
    )
    parser.add_argument(
        "-d",
        "--data-load-path",
        type=str,
        dest="data_load_path",
        default=DEFAULT_DATA_PATH,
        help=f"relative path to load test data from (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "-f",
        "--file-mode",
        type=bool,
        dest="file_mode",
        default=False,
        help="if true interprets load path as file, otherwise - as folder. \
          (default: False)",
    )
    parser.add_argument(
        "-c",
        "--cuda",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="use CUDA if available (default: True)",
    )
    parser.add_argument(
        "-w",
        "--ignore-warnings",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="ignore warnings (default: True)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="print information (default: True)",
    )

    namespace = parser.parse_args()
    (
        model_path,
        save_path,
        data_load_path,
        file_mode,
        cuda,
        ignore_warnings,
        verbose,
    ) = (
        namespace.model,
        namespace.save_path,
        namespace.data_load_path,
        namespace.file_mode,
        namespace.cuda,
        namespace.ignore_warnings,
        namespace.verbose,
    )
    file_mode: bool = bool(file_mode)
    cuda: bool = bool(cuda)
    ignore_warnings: bool = bool(ignore_warnings)
    verbose: bool = bool(verbose)

    if ignore_warnings:
        warnings.filterwarnings("ignore")

    if cuda:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up logger
    logger = Logger(verbose)

    model = load_model(model_path, logger)
    dataset = load_and_build_dataset(data_load_path, file_mode, logger)

    logger.log("Done!")


if __name__ == "__main__":
    evaluate()
