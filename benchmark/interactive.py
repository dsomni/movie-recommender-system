""" Module contains functions for interaction with models """

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

""" Constants """

# Default paths
DEFAULT_MODEL_PATH = "./models/bce_masks_split"
DEFAULT_ML_100K_PATH = "./data/raw/ml-100k/"

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


""" Interactive mode functions"""


def load_genres(path: str) -> list[str]:
    return pd.read_csv(
        os.path.join(path, "u.genre"),
        sep="|",
        header=None,
        names=["name", "genre_idx"],
        encoding="ISO-8859-1",
    )["name"].tolist()


def load_items(path: str, genres: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(path, "u.item"),
        sep="|",
        header=None,
        names=[
            "movie_id",
            "movie_title",
            "release_date",
            "video_release_date",
            "IMDb_URL",
            *genres,
        ],
        encoding="ISO-8859-1",
    )


def calculate_genre_ratios(
    movie_indices: np.ndarray, items_df: pd.DataFrame
) -> np.ndarray:
    genres_sum = (
        items_df[items_df["movie_id"].isin(movie_indices + 1)]
        .iloc[:, 5:]
        .sum(axis=0)
        .to_numpy()
    )
    return genres_sum / genres_sum.sum()


def get_recommendations(
    model: nn.Module,
    encoded_age: float,
    encoded_gender: int,
    encoded_occupation: int,
    movie_indices: list[int],
    movies_df: pd.DataFrame,
    predicted_threshold: float,
    num_recs: int,
) -> np.ndarray:
    movie_indices_shifted = np.array(movie_indices) - 1  # starting from 0

    movies_ratings = np.zeros(NUM_MOVIES)
    movies_ratings[movie_indices_shifted] = 1.0  # rating = 5
    input_vector = np.array(
        [
            encoded_age,
            encoded_gender,
            encoded_occupation,
            *calculate_genre_ratios(np.array(movie_indices_shifted), movies_df),
            *movies_ratings,
        ]
    )

    predictions = get_single_output(model, input_vector)
    predictions[predictions < predicted_threshold] = 0.0
    unseen_predictions = get_unseen_on_input_data(movies_ratings, predictions)

    movie_ids = np.argsort(-unseen_predictions) + 1

    unknown_idx = 267  # actual idx (from 1)
    movie_ids = np.delete(movie_ids, np.where(movie_ids == unknown_idx))

    return movie_ids[:num_recs]


def get_movie_titles(
    recommended_movies: np.ndarray, movies_df: pd.DataFrame
) -> list[str]:
    return [
        movies_df[movies_df["movie_id"] == movie_id]["movie_title"].to_list()[0]
        for movie_id in recommended_movies
    ]


def print_titles(movie_titles: list[str]):
    print("Model suggest you to check out the following movies: ")
    for idx, movie_title in enumerate(movie_titles):
        print(f"{(idx+1):2}\t{movie_title}")
    print()


""" Main interaction function """


def interact():
    """Interact with model"""
    global DEVICE

    # Parse arguments
    parser = argparse.ArgumentParser(description="Interact with given model")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        dest="model",
        default=DEFAULT_MODEL_PATH,
        help="path to pytorch model",
    )

    parser.add_argument(
        "-a",
        "--age",
        type=int,
        dest="age",
        default=21,
        help="age (default: 21)",
    )
    parser.add_argument(
        "-g",
        "--gender",
        choices=[0, 1],
        dest="gender",
        default=1,
        help="gender. 1 for male and 0 for female (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--occupation",
        type=int,
        dest="occupation",
        default=19,
        help="occupation index (from 1) from ml-100k dataset (default: 19 - student)",
    )
    parser.add_argument(
        "-f",
        "--favorite-movies",
        type=int,
        nargs="*",
        dest="movies",
        default=[56],
        help="movie indices (from 1) from ml-100k dataset \
            (default: [56] -  Pulp fiction)",
    )

    parser.add_argument(
        "-n",
        "--num-recs",
        type=int,
        dest="num_recs",
        default=5,
        help="how many movies to recommend (default: 5)",
    )
    parser.add_argument(
        "-p",
        "--predicted-threshold",
        type=float,
        dest="predicted_threshold",
        default=0.0,
        help="threshold for predicted ratings from 0 to 1 (default: 0)",
    )
    parser.add_argument(
        "-r",
        "--raw-ml100k-data-path",
        type=str,
        dest="ml_100k_path",
        default=DEFAULT_ML_100K_PATH,
        help=f"relative path to raw ML 100k data (default: {DEFAULT_ML_100K_PATH})",
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
        age,
        gender,
        occupation,
        movies,
        num_recs,
        predicted_threshold,
        ml_100k_path,
        cuda,
        ignore_warnings,
        verbose,
    ) = (
        namespace.model,
        namespace.age,
        namespace.gender,
        namespace.occupation,
        namespace.movies,
        namespace.num_recs,
        namespace.predicted_threshold,
        namespace.ml_100k_path,
        namespace.cuda,
        namespace.ignore_warnings,
        namespace.verbose,
    )
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

    logger.log("Loading genres...")
    genres = load_genres(ml_100k_path)
    logger.log("Success!")

    logger.log("Loading movies...")
    movies_df = load_items(ml_100k_path, genres)
    logger.log("Success!")

    logger.log("Generating recommendations...")
    recommended_movies = get_recommendations(
        model,
        age / 100,
        gender,
        occupation,
        movies,
        movies_df,
        predicted_threshold,
        num_recs,
    )
    logger.log("Success!")

    logger.log("Mapping movie titles...")
    movie_titles = get_movie_titles(recommended_movies, movies_df)
    logger.log("Success!")

    print_titles(movie_titles)

    logger.log("Done!")


if __name__ == "__main__":
    interact()
