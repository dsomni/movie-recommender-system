""" Module contains functions for evaluating models """

import argparse
import json
import os
import time
import warnings
from ast import literal_eval
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from torcheval.metrics.functional.ranking import retrieval_precision
from tqdm import tqdm

""" Constants """

# Default paths
DEFAULT_MODEL_PATH = "./models/tunned_bce_msplit_latest"
DEFAULT_DATA_PATH = "./benchmark/data/masks_split/test"
DEFAULT_SAVE_PATH = "./benchmark/data/generated/"

# Data-specific constants
NUM_MOVIES = 1682
BASIC_USER_FEATURES = 3
TOTAL_USER_FEATURES = BASIC_USER_FEATURES + 19

# Model constants
INPUT_SIZE = TOTAL_USER_FEATURES + NUM_MOVIES
DEVICE = torch.device("cpu")

# Metrics constants
DEFAULT_METRICS_KS = [5, 10, 20, 50]


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
    """Manage recommendation entities data"""

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
    """Torch Recommendation system model"""

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
        """Forward pass

        Args:
            x (Any): input data

        Returns:
            Any: model output data
        """
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.relu(self.fc3(x))

        return F.sigmoid(self.fc4(x))


""" Save & load functions """


def load_model(model_path: str, logger: Logger) -> nn.Module:
    """Load model from disk

    Args:
        model_path (str): path to load model from
        logger (Logger): logger instance

    Returns:
        nn.Module: torch model
    """
    logger.log(f"Loading model from '{model_path}'...")
    model = torch.load(model_path)
    model.eval()
    logger.log("Success!")
    return model


def load_and_build_dataset(
    data_path: str, file_mode: bool, logger: Logger
) -> RecommendationDataset:
    """Load data from disk and build dataset

    Args:
        data_path (str): path to load data from
        file_mode (bool): should interpret data_path as file or not
        logger (Logger): logger instance

    Returns:
        RecommendationDataset: dataset
    """
    if file_mode:
        logger.log(f"File mode is ON. Reading '{data_path}' as file...")
        df = pd.read_csv(data_path)
    else:
        logger.log(f"File mode is OFF. Reading from folder '{data_path}'...")
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


def save_metrics(metrics_dict: dict, path: str, model_name: str, logger: Logger):
    """Save metrics dictionary to disk

    Args:
        metrics_dict (dict): metrics information
        path (str): path to save data to
        model_name (str): unique model name for file name
        logger (Logger): logger instance
    """
    full_path = os.path.join(path, f"{model_name}.json")
    logger.log(f"Saving metrics to '{full_path}'...")
    with open(full_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    logger.log("Success!")


""" Model functions """


def get_unseen_on_input_data(
    input_ratings: np.ndarray, movie_ratings: np.ndarray
) -> np.ndarray:
    """Return only movies which are unseen on input data

    Args:
        input_ratings (np.ndarray): input ratings vector
        movie_ratings (np.ndarray): output ratings vector

    Returns:
        np.ndarray: unseen movies indices
    """
    unseen_ratings = movie_ratings.copy()
    seen_indices = np.nonzero(input_ratings > 0)[0]
    unseen_ratings[seen_indices] = 0
    return unseen_ratings


def get_single_output(
    model: nn.Module,
    input_data: np.ndarray,
) -> np.ndarray:
    """Perform single model call

    Args:
        model (nn.Module): torch model to call
        input_data (np.ndarray): input data vector

    Returns:
        np.ndarray: model output
    """
    with torch.no_grad():
        model.eval()
        input_tensor = torch.Tensor([input_data]).to(DEVICE)
        model_out = model(input_tensor)

    return model_out[0].cpu().numpy()


def generate_test_data(
    model: nn.Module, dataset: RecommendationDataset, logger: Logger
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collect inputs together with model outputs

    Args:
        model (nn.Module): torch model to call
        dataset (RecommendationDataset): dataset to generate data on
        logger (Logger): logger instance

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: list of (input, output) pairs
    """
    logger.log("Generating test data...")
    test_data = []

    loop = dataset
    if logger.verbose:
        loop = tqdm(loop)

    for input_data, _, target in loop:
        predicted = get_single_output(model, input_data)

        input_ratings = input_data[TOTAL_USER_FEATURES:]
        unseen_predicted = get_unseen_on_input_data(input_ratings, predicted)
        unseen_target = get_unseen_on_input_data(input_ratings, target)
        test_data.append((unseen_target, unseen_predicted))

    logger.log("Success!")
    return test_data


""" Metrics calculation """


def get_top_args(x: np.ndarray, n: int) -> np.ndarray:
    """Get indices of Top-N values

    Args:
        x (np.ndarray): values
        n (int): how many to return

    Returns:
        np.ndarray: indices of Top-N values
    """
    return np.argsort(-x)[:n]


def top_intersection(target: np.ndarray, predicted: np.ndarray, top_n: int) -> list[int]:
    """Calculates top intersection sets

    Args:
        target (np.ndarray): target indices
        predicted (np.ndarray): predicted indices
        top_n (int): how many to take from top

    Returns:
        list[int]: indices of Top-N intersections
    """
    return list(
        set(get_top_args(target, top_n)).intersection(get_top_args(predicted, top_n))
    )


def top_k_intersections(
    data: list[tuple[np.ndarray, np.ndarray]], k: int, threshold: float
) -> list[int]:
    """Calculates top k intersection sets length

    Args:
        data (list[tuple[np.ndarray, np.ndarray]]): list of (target, predicted) pairs
        k (int): how many to take from top
        threshold (float): threshold for rating to take corresponding movie


    Returns:
        list[int]: indices of intersection sets length
    """
    intersections = []
    for unseen_target, unseen_predicted in data:
        nonzero_targets = unseen_target[unseen_target > threshold]
        relevant_predicted = unseen_predicted[unseen_predicted > threshold]
        intersections.append(
            len(top_intersection(nonzero_targets, relevant_predicted, k))
        )

    return intersections


def retrieval_precisions_on_k(
    data: list[tuple[np.ndarray, np.ndarray]], k: int, threshold: float
) -> list[int]:
    """Calculate retrieval precisions (from torch) on k

    Args:
        data (list[tuple[np.ndarray, np.ndarray]]): list of (target, predicted) pairs
        k (int): how many items to inspect
        threshold (float): threshold for rating to take corresponding movie

    Returns:
        list[int]: retrieval precisions on data
    """
    retrieval_precisions = []
    for unseen_target, unseen_predicted in data:
        nonzero_targets = unseen_target > threshold
        relevant_predicted = unseen_predicted

        retrieval_precisions.append(
            retrieval_precision(
                torch.Tensor(relevant_predicted), torch.Tensor(nonzero_targets), k
            ).item()
        )

    return retrieval_precisions


def precision_scores(
    data: list[tuple[np.ndarray, np.ndarray]], threshold: float
) -> list[int]:
    """Calculate precision scores (from sklearn)

    Args:
        data (list[tuple[np.ndarray, np.ndarray]]): list of (target, predicted) pairs
        threshold (float): threshold for rating to take corresponding movie

    Returns:
        list[int]: precision scores on data
    """
    precisions = []
    for unseen_target, unseen_predicted in data:
        nonzero_targets = unseen_target > threshold
        relevant_predicted = unseen_predicted > threshold

        precisions.append(precision_score(relevant_predicted, nonzero_targets))

    return precisions


def recall_scores(
    data: list[tuple[np.ndarray, np.ndarray]], threshold: float
) -> list[int]:
    """Calculate recall scores (from sklearn)

    Args:
        data (list[tuple[np.ndarray, np.ndarray]]): list of (target, predicted) pairs
        threshold (float): threshold for rating to take corresponding movie

    Returns:
        list[int]: recall scores on data
    """
    recalls = []
    for unseen_target, unseen_predicted in data:
        nonzero_targets = unseen_target > threshold
        relevant_predicted = unseen_predicted > threshold

        recalls.append(recall_score(relevant_predicted, nonzero_targets))

    return recalls


def average_precision_on_k(target: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """Calculate AP@K

    Args:
        target (np.ndarray): target indices
        predicted (np.ndarray): predicted indices
        k (int): how many items to inspect

    Returns:
        float: AP@K score
    """
    relevant_predicted = predicted.copy()
    if len(relevant_predicted) > k:
        relevant_predicted = relevant_predicted[:k]

    score = 0.0
    hits = 0

    for idx, x in enumerate(relevant_predicted):
        if x in target and x not in relevant_predicted[:idx]:
            hits += 1
            score += hits / (idx + 1.0)

    return score / max(min(len(target), k), 1)


def map_on_k(
    targets: list[np.ndarray], predictions: list[np.ndarray], k: int
) -> tuple[float, list[float]]:
    """Calculate MAP@K

    Args:
        targets (list[np.ndarray]): list of target indices
        predictions (list[np.ndarray]): list of predicted indices
        k (int): how many items to inspect

    Returns:
        tuple[float, list[float]]:  (MAP@K score, list of AP@K scores)
    """
    average_precisions = [
        average_precision_on_k(target, predicted, k)
        for target, predicted in zip(targets, predictions)
    ]
    return np.mean(average_precisions), average_precisions


def generate_total_data_lists(
    data: list[tuple[np.ndarray, np.ndarray]], threshold: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Concatenate all inputs and outputs into accumulated lists

    Args:
        data (list[tuple[np.ndarray, np.ndarray]]): list of (target, predicted) pairs
        threshold (float): threshold for rating to take corresponding movie

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]:
        (list of accumulated inputs, list of accumulated outputs)
    """
    all_targets = []
    all_predictions = []
    for unseen_target, unseen_predicted in data:
        nonzero_targets = unseen_target > threshold
        all_targets.append(
            np.argsort(nonzero_targets)[len(nonzero_targets) - sum(nonzero_targets) :]
        )
        all_predictions.append(np.argsort(-unseen_predicted))

    return all_targets, all_predictions


def get_metrics_dict(
    data: list[tuple[np.ndarray, np.ndarray]],
    ks: list[int],
    threshold: float,
    logger: Logger,
) -> dict:
    """Build metrics dictionary

    Args:
        data (list[tuple[np.ndarray, np.ndarray]]): list of (target, predicted) pairs
        ks (list[int]): list of k to inspect
        threshold (float): threshold for rating to take corresponding movie
        logger (Logger): logger instance

    Returns:
        dict: metrics dictionary
    """
    logger.log("Generating metrics...")
    all_targets, all_predictions = generate_total_data_lists(data, threshold=threshold)
    precisions = precision_scores(data, threshold=threshold)
    recalls = recall_scores(data, threshold=threshold)
    metrics_dict: dict[str, Any] = {
        "global": {
            "mean_precision": float(np.mean(precisions)),
            "mean_recalls": float(np.mean(recalls)),
        }
    }
    for k in ks:
        intersections = top_k_intersections(data, k, threshold=threshold)
        retrieval_precisions = retrieval_precisions_on_k(data, k, threshold=threshold)
        map_score, average_precisions = map_on_k(all_targets, all_predictions, k)

        metrics_dict[str(k)] = {
            "map": float(map_score),
            "mean_retrieval_precision": float(np.mean(retrieval_precisions)),
            "mean_top_intersections": float(np.mean(intersections)),
            "average_precisions": average_precisions,
            "retrieval_precisions": retrieval_precisions,
            "intersections": intersections,
        }
    logger.log("Success!")
    return metrics_dict


""" Visualization functions"""


def save_metrics_plot(
    model_name: str,
    plot_title: str,
    metrics_dict: dict,
    save_path: str,
    logger: Logger,
    figsize=(10, 12),
):
    """Save metrics plot as png file

    Args:
        model_name (str): unique model name for file name
        plot_title (str): title of plot
        metrics_dict (dict): metrics dictionary
        save_path (str): path to save plot picture
        logger (Logger): logger instance
        figsize (Any): plot figure size. Default: (10, 12)

    Returns:
        dict: metrics dictionary
    """
    num_colors = len(metrics_dict) - 1
    cm = plt.get_cmap("gist_rainbow")
    colors = [cm(1.0 * i / num_colors) for i in range(num_colors)]

    k_values = list(filter(lambda x: x != "global", metrics_dict.keys()))

    fig, axs = plt.subplots(4, 1, figsize=figsize)
    ax1, ax2, ax3, ax4 = axs.flat

    fig.suptitle(plot_title, fontsize=10)
    ax1.set_title("MAP@K")
    ax2.set_title("MRP@K")
    ax3.set_title("Average precisions")
    ax4.set_title("Retrieval Precisions")

    for ax in (ax1, ax2):
        ax.set(xlabel="K", ylabel="")

    for ax in axs.flat:
        ax.grid()
        ax.set_prop_cycle(color=colors)

    data_points = list(range(len(metrics_dict[k_values[0]]["average_precisions"])))

    maps = []
    mean_retrieval_precisions = []

    for k_value in k_values:
        metrics = metrics_dict[k_value]
        maps.append(metrics["map"])
        mean_retrieval_precisions.append(metrics["mean_retrieval_precision"])

        ax3.scatter(data_points, metrics["average_precisions"], s=4, label=f"K={k_value}")
        ax4.scatter(
            data_points, metrics["retrieval_precisions"], s=4, label=f"K={k_value}"
        )

    ax1.bar(k_values, maps, color=colors)
    ax1.get_xaxis().set_visible(False)
    ax2.bar(k_values, mean_retrieval_precisions, color=colors)
    ax2.get_xaxis().set_visible(False)

    lines_labels = [ax3.get_legend_handles_labels()]
    lines, labels = [sum(x, []) for x in zip(*lines_labels)]
    fig.legend(
        lines,
        labels,
        scatterpoints=1,
        markerscale=3,
        loc="outside lower center",
        ncol=min(6, len(k_values)),
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()
    full_path = os.path.join(save_path, f"{model_name}.png")
    logger.log(f"Saving plot to '{full_path}'...")
    plt.savefig(full_path, bbox_inches="tight")
    logger.log("Success!")
    plt.close()


""" Main evaluation function """


def evaluate():
    """Evaluate model"""
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
        "-k",
        "--metric-ks",
        type=int,
        nargs="+",
        dest="metric_ks",
        default=DEFAULT_METRICS_KS,
        help=f"k values for some metrics (like MAP@K) \
            (default: {DEFAULT_METRICS_KS})",
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
        "-s",
        "--save-path",
        type=str,
        dest="save_path",
        default=DEFAULT_SAVE_PATH,
        help=f"relative path to save generated results (default: {DEFAULT_SAVE_PATH})",
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
        dest="file_mode",
        default=False,
        action=argparse.BooleanOptionalAction,
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
        metric_ks,
        predicted_threshold,
        save_path,
        data_load_path,
        file_mode,
        cuda,
        ignore_warnings,
        verbose,
    ) = (
        namespace.model,
        namespace.metric_ks,
        namespace.predicted_threshold,
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

    logger = Logger(verbose)

    model = load_model(model_path, logger)
    dataset = load_and_build_dataset(data_load_path, file_mode, logger)

    model_name = os.path.split(model_path)[1]
    model_metrics_name = f"{model_name}_{int(time.time()*1000)}"

    test_data = generate_test_data(model, dataset, logger)

    metrics_dict = get_metrics_dict(test_data, metric_ks, predicted_threshold, logger)

    save_metrics(metrics_dict, save_path, model_metrics_name, logger)

    save_metrics_plot(
        model_metrics_name,
        f"{model_name} on {data_load_path}",
        metrics_dict,
        save_path,
        logger,
    )

    logger.log("Done!")


if __name__ == "__main__":
    evaluate()
