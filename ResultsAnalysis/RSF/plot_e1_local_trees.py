"""
RSF E1: effect of the number of local trees on mean performance.

Design:
- X-axis: local trees (20, 50, 100, 200)
- Y-axis: mean performance across clients
- Type: line plot with one subplot per metric
"""

from __future__ import annotations

import math
import os
from pathlib import Path

MPL_CONFIG_DIR = Path(__file__).with_name(".mplconfig")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np


plt.style.use("seaborn-v0_8-whitegrid")

OUTPUT_PATH = Path(__file__).with_name("figure_1_E1_RSF.png")
LOCAL_TREES = [20, 50, 100, 200]

METRIC_STYLES = {
    "c_index": {"label": "C-index", "color": "#0072B2", "marker": "o"},
    "auc": {"label": "AUC", "color": "#009E73", "marker": "s"},
    "ibs": {"label": "IBS", "color": "#D55E00", "marker": "^"},
}

TITLE_FONT_SIZE = 17
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 11.5


def metric_entry(means: list[float], stds: list[float]) -> dict[str, list[float]]:
    return {"means": means, "stds": stds}


RAW_RESULTS = {
    20: {
        "c_index": metric_entry([0.823, 0.497, 0.689, 0.634, 0.936], [0.029, 0.199, 0.070, 0.097, 0.058]),
        "auc": metric_entry([0.772, 0.453, 0.533, 0.702, 0.939], [0.043, 0.207, 0.139, 0.081, 0.028]),
        "ibs": metric_entry([0.164, 0.191, 0.192, 0.094, 0.047], [0.007, 0.016, 0.010, 0.006, 0.001]),
    },
    50: {
        "c_index": metric_entry([0.831, 0.536, 0.670, 0.666, 0.948], [0.023, 0.207, 0.061, 0.082, 0.033]),
        "auc": metric_entry([0.785, 0.469, 0.503, 0.717, 0.945], [0.039, 0.208, 0.119, 0.061, 0.030]),
        "ibs": metric_entry([0.163, 0.188, 0.193, 0.096, 0.046], [0.007, 0.019, 0.010, 0.006, 0.001]),
    },
    100: {
        "c_index": metric_entry([0.811, 0.515, 0.675, 0.592, 0.948], [0.030, 0.144, 0.053, 0.096, 0.063]),
        "auc": metric_entry([0.770, 0.449, 0.509, 0.680, 0.957], [0.036, 0.146, 0.066, 0.065, 0.020]),
        "ibs": metric_entry([0.163, 0.189, 0.191, 0.096, 0.046], [0.006, 0.018, 0.012, 0.006, 0.001]),
    },
    200: {
        "c_index": metric_entry([0.823, 0.511, 0.666, 0.596, 0.931], [0.021, 0.110, 0.074, 0.097, 0.054]),
        "auc": metric_entry([0.769, 0.431, 0.482, 0.665, 0.947], [0.028, 0.121, 0.061, 0.082, 0.036]),
        "ibs": metric_entry([0.161, 0.194, 0.192, 0.097, 0.047], [0.007, 0.019, 0.011, 0.006, 0.001]),
    },
}


def aggregate_metric(metric_data: dict[str, list[float]]) -> tuple[float, float]:
    means = metric_data["means"]
    stds = metric_data["stds"]
    mean_value = float(np.mean(means))
    propagated_std = math.sqrt(sum(std * std for std in stds)) / len(stds)
    return mean_value, propagated_std


def build_series(metric_name: str) -> tuple[list[float], list[float]]:
    mean_series = []
    std_series = []
    for local_trees in LOCAL_TREES:
        mean_value, std_value = aggregate_metric(RAW_RESULTS[local_trees][metric_name])
        mean_series.append(mean_value)
        std_series.append(std_value)
    return mean_series, std_series


def add_value_labels(ax: plt.Axes, x_values: list[int], y_values: list[float], color: str) -> None:
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.03
    for x_value, y_value in zip(x_values, y_values):
        ax.annotate(
            f"{y_value:.2f}",
            xy=(x_value, y_value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
            color=color,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 0.2,
            },
        )


def plot_e1_local_trees() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))

    for ax, metric_name in zip(axes, ["c_index", "auc", "ibs"]):
        style = METRIC_STYLES[metric_name]
        mean_series, std_series = build_series(metric_name)

        ax.errorbar(
            LOCAL_TREES,
            mean_series,
            yerr=std_series,
            color=style["color"],
            marker=style["marker"],
            markersize=7,
            linewidth=2.2,
            capsize=4,
        )
        add_value_labels(ax, LOCAL_TREES, mean_series, style["color"])

        ax.set_title(style["label"], fontsize=TITLE_FONT_SIZE, fontweight="bold")
        ax.set_xlabel("Local trees", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Mean performance", fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(LOCAL_TREES)
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        lower_limit = min(mean - std for mean, std in zip(mean_series, std_series))
        upper_limit = max(mean + std for mean, std in zip(mean_series, std_series))
        padding = 0.06 if metric_name != "ibs" else 0.03
        ax.set_ylim(lower_limit - padding, upper_limit + padding)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

    print(f"Saved figure to: {OUTPUT_PATH}")
    for metric_name in ["c_index", "auc", "ibs"]:
        mean_series, _ = build_series(metric_name)
        print(f"{METRIC_STYLES[metric_name]['label']}: {dict(zip(LOCAL_TREES, [round(value, 3) for value in mean_series]))}")


if __name__ == "__main__":
    plot_e1_local_trees()
