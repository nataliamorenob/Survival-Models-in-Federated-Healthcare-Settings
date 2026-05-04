"""
RSF E2 main comparison plot.

Design:
- X-axis: number of clients (3, 4, 5)
- Bars within each group: Local, Federated, Centralized
- Y-axis: performance
- Subplots: C-index, AUC, IBS

For each training paradigm, E2 reports results for tree settings 50, 100, 200.
This script uses the fixed 100-tree configuration for Local, Federated, and
Centralized across all metrics and client counts.
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
from matplotlib.patches import Patch


plt.style.use("seaborn-v0_8-whitegrid")

OUTPUT_PATH = Path(__file__).with_name("figure_2_E2_RSF.png")
FIXED_LOCAL_TREE = 100
FIXED_FED_TREE = 100
FIXED_CENTRALIZED_TREE = 100

PARADIGMS = ["Local", "Federated", "Centralized"]
CLIENT_COUNTS = [5, 4, 3]

LOCAL_COLOR = "#D55E00"  # reddish-orange
FEDERATED_COLOR = "#0072B2"  # blue
CENTRALIZED_COLOR = "#009E73"  # bluish-green

COLORS = {
    "Local": LOCAL_COLOR,
    "Federated": FEDERATED_COLOR,
    "Centralized": CENTRALIZED_COLOR,
}

TITLE_FONT_SIZE = 17
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 11.5
LEGEND_FONT_SIZE = 13


def metric_entry(means: list[float], stds: list[float]) -> dict[str, list[float]]:
    return {"means": means, "stds": stds}


RAW_RESULTS = {
    5: {
        "Local": {
            50: {
                "c_index": metric_entry([0.827, 0.365, 0.657, 0.587, 0.978], [0.011, 0.156, 0.093, 0.085, 0.031]),
                "auc": metric_entry([0.753, 0.374, 0.636, 0.615, 0.950], [0.031, 0.137, 0.110, 0.078, 0.023]),
                "ibs": metric_entry([0.150, 0.283, 0.187, 0.097, 0.042], [0.011, 0.050, 0.011, 0.010, 0.001]),
            },
            100: {
                "c_index": metric_entry([0.824, 0.368, 0.658, 0.608, 0.978], [0.015, 0.098, 0.090, 0.103, 0.031]),
                "auc": metric_entry([0.745, 0.374, 0.643, 0.613, 0.951], [0.036, 0.118, 0.102, 0.104, 0.021]),
                "ibs": metric_entry([0.150, 0.283, 0.189, 0.096, 0.043], [0.010, 0.047, 0.011, 0.010, 0.001]),
            },
            200: {
                "c_index": metric_entry([0.825, 0.368, 0.625, 0.608, 0.978], [0.013, 0.128, 0.077, 0.091, 0.031]),
                "auc": metric_entry([0.743, 0.377, 0.636, 0.613, 0.954], [0.029, 0.130, 0.094, 0.076, 0.019]),
                "ibs": metric_entry([0.150, 0.284, 0.191, 0.096, 0.042], [0.012, 0.039, 0.010, 0.009, 0.001]),
            },
        },
        "Federated": {
            50: {
                "c_index": metric_entry([0.820, 0.563, 0.668, 0.638, 0.929], [0.028, 0.172, 0.063, 0.061, 0.081]),
                "auc": metric_entry([0.760, 0.484, 0.472, 0.654, 0.910], [0.041, 0.155, 0.092, 0.133, 0.087]),
                "ibs": metric_entry([0.163, 0.191, 0.193, 0.095, 0.047], [0.009, 0.022, 0.012, 0.007, 0.001]),
            },
            100: {
                "c_index": metric_entry([0.821, 0.520, 0.684, 0.658, 0.938], [0.020, 0.130, 0.057, 0.070, 0.051]),
                "auc": metric_entry([0.765, 0.443, 0.512, 0.731, 0.937], [0.026, 0.146, 0.092, 0.046, 0.032]),
                "ibs": metric_entry([0.163, 0.191, 0.191, 0.096, 0.047], [0.007, 0.018, 0.010, 0.006, 0.001]),
            },
            200: {
                "c_index": metric_entry([0.816, 0.511, 0.685, 0.627, 0.925], [0.026, 0.136, 0.036, 0.065, 0.049]),
                "auc": metric_entry([0.766, 0.448, 0.495, 0.697, 0.941], [0.028, 0.139, 0.057, 0.061, 0.032]),
                "ibs": metric_entry([0.163, 0.189, 0.192, 0.096, 0.047], [0.006, 0.015, 0.010, 0.006, 0.0008]),
            },
        },
        "Centralized": {
            50: {
                "c_index": metric_entry([0.823, 0.600, 0.754, 0.681, 0.906], [0.012, 0.123, 0.052, 0.087, 0.055]),
                "auc": metric_entry([0.765, 0.575, 0.640, 0.746, 0.879], [0.030, 0.095, 0.089, 0.082, 0.040]),
                "ibs": metric_entry([0.153, 0.789, 0.141, 0.125, 0.039], [0.007, 0.150, 0.008, 0.009, 0.001]),
            },
            100: {
                "c_index": metric_entry([0.823, 0.600, 0.744, 0.670, 0.895], [0.013, 0.108, 0.044, 0.071, 0.066]),
                "auc": metric_entry([0.765, 0.578, 0.623, 0.749, 0.885], [0.030, 0.087, 0.075, 0.065, 0.033]),
                "ibs": metric_entry([0.154, 0.766, 0.143, 0.124, 0.039], [0.006, 0.119, 0.006, 0.009, 0.001]),
            },
            200: {
                "c_index": metric_entry([0.825, 0.611, 0.746, 0.681, 0.910], [0.014, 0.087, 0.051, 0.084, 0.065]),
                "auc": metric_entry([0.768, 0.579, 0.618, 0.750, 0.894], [0.031, 0.080, 0.082, 0.061, 0.022]),
                "ibs": metric_entry([0.153, 0.770, 0.143, 0.125, 0.039], [0.006, 0.111, 0.006, 0.010, 0.001]),
            },
        },
    },
    4: {
        "Local": {
            50: {
                "c_index": metric_entry([0.827, 0.365, 0.657, 0.587], [0.011, 0.156, 0.093, 0.085]),
                "auc": metric_entry([0.753, 0.374, 0.636, 0.615], [0.031, 0.137, 0.110, 0.078]),
                "ibs": metric_entry([0.150, 0.283, 0.187, 0.097], [0.011, 0.050, 0.011, 0.010]),
            },
            100: {
                "c_index": metric_entry([0.824, 0.368, 0.658, 0.608], [0.015, 0.098, 0.090, 0.103]),
                "auc": metric_entry([0.745, 0.374, 0.643, 0.613], [0.036, 0.118, 0.102, 0.104]),
                "ibs": metric_entry([0.150, 0.283, 0.189, 0.096], [0.010, 0.047, 0.011, 0.010]),
            },
            200: {
                "c_index": metric_entry([0.825, 0.368, 0.625, 0.608], [0.013, 0.128, 0.077, 0.091]),
                "auc": metric_entry([0.743, 0.377, 0.636, 0.613], [0.029, 0.130, 0.094, 0.076]),
                "ibs": metric_entry([0.150, 0.284, 0.191, 0.096], [0.012, 0.039, 0.010, 0.009]),
            },
        },
        "Federated": {
            50: {
                "c_index": metric_entry([0.788, 0.570, 0.694, 0.647], [0.064, 0.198, 0.048, 0.102]),
                "auc": metric_entry([0.735, 0.489, 0.533, 0.733], [0.045, 0.177, 0.087, 0.108]),
                "ibs": metric_entry([0.162, 0.195, 0.188, 0.094], [0.007, 0.018, 0.011, 0.007]),
            },
            100: {
                "c_index": metric_entry([0.804, 0.511, 0.730, 0.668], [0.020, 0.146, 0.028, 0.043]),
                "auc": metric_entry([0.756, 0.468, 0.583, 0.753], [0.026, 0.162, 0.078, 0.043]),
                "ibs": metric_entry([0.161, 0.194, 0.184, 0.096], [0.005, 0.016, 0.010, 0.007]),
            },
            200: {
                "c_index": metric_entry([0.811, 0.495, 0.730, 0.664], [0.018, 0.178, 0.021, 0.078]),
                "auc": metric_entry([0.760, 0.433, 0.555, 0.761], [0.027, 0.169, 0.057, 0.046]),
                "ibs": metric_entry([0.162, 0.196, 0.187, 0.094], [0.006, 0.020, 0.010, 0.006]),
            },
        },
        "Centralized": {
            50: {
                "c_index": metric_entry([0.817, 0.520, 0.777, 0.704], [0.017, 0.091, 0.028, 0.078]),
                "auc": metric_entry([0.762, 0.525, 0.657, 0.778], [0.022, 0.107, 0.070, 0.053]),
                "ibs": metric_entry([0.137, 0.707, 0.131, 0.103], [0.005, 0.098, 0.008, 0.008]),
            },
            100: {
                "c_index": metric_entry([0.824, 0.559, 0.786, 0.718], [0.015, 0.075, 0.037, 0.058]),
                "auc": metric_entry([0.768, 0.520, 0.675, 0.789], [0.021, 0.071, 0.065, 0.046]),
                "ibs": metric_entry([0.137, 0.715, 0.130, 0.101], [0.005, 0.089, 0.007, 0.007]),
            },
            200: {
                "c_index": metric_entry([0.827, 0.565, 0.799, 0.727], [0.013, 0.052, 0.040, 0.063]),
                "auc": metric_entry([0.771, 0.550, 0.687, 0.795], [0.022, 0.077, 0.076, 0.046]),
                "ibs": metric_entry([0.137, 0.719, 0.130, 0.101], [0.005, 0.096, 0.007, 0.008]),
            },
        },
    },
    3: {
        "Local": {
            50: {
                "c_index": metric_entry([0.827, 0.365, 0.657], [0.011, 0.156, 0.093]),
                "auc": metric_entry([0.753, 0.374, 0.636], [0.031, 0.137, 0.110]),
                "ibs": metric_entry([0.150, 0.283, 0.187], [0.011, 0.050, 0.011]),
            },
            100: {
                "c_index": metric_entry([0.824, 0.368, 0.658], [0.015, 0.098, 0.090]),
                "auc": metric_entry([0.745, 0.374, 0.643], [0.036, 0.118, 0.102]),
                "ibs": metric_entry([0.150, 0.283, 0.189], [0.010, 0.047, 0.011]),
            },
            200: {
                "c_index": metric_entry([0.825, 0.368, 0.625], [0.013, 0.128, 0.077]),
                "auc": metric_entry([0.743, 0.377, 0.636], [0.029, 0.130, 0.094]),
                "ibs": metric_entry([0.150, 0.284, 0.191], [0.012, 0.039, 0.010]),
            },
        },
        "Federated": {
            50: {
                "c_index": metric_entry([0.823, 0.534, 0.700], [0.017, 0.090, 0.037]),
                "auc": metric_entry([0.768, 0.427, 0.555], [0.030, 0.094, 0.085]),
                "ibs": metric_entry([0.157, 0.201, 0.187], [0.009, 0.028, 0.013]),
            },
            100: {
                "c_index": metric_entry([0.807, 0.554, 0.708], [0.011, 0.127, 0.046]),
                "auc": metric_entry([0.749, 0.475, 0.575], [0.034, 0.133, 0.085]),
                "ibs": metric_entry([0.159, 0.197, 0.186], [0.007, 0.021, 0.010]),
            },
            200: {
                "c_index": metric_entry([0.818, 0.515, 0.710], [0.019, 0.075, 0.038]),
                "auc": metric_entry([0.762, 0.443, 0.577], [0.027, 0.099, 0.089]),
                "ibs": metric_entry([0.158, 0.199, 0.184], [0.006, 0.020, 0.010]),
            },
        },
        "Centralized": {
            50: {
                "c_index": metric_entry([0.834, 0.579, 0.765], [0.019, 0.130, 0.043]),
                "auc": metric_entry([0.768, 0.503, 0.652], [0.025, 0.112, 0.072]),
                "ibs": metric_entry([0.141, 0.677, 0.133], [0.004, 0.107, 0.005]),
            },
            100: {
                "c_index": metric_entry([0.832, 0.579, 0.762], [0.018, 0.120, 0.044]),
                "auc": metric_entry([0.770, 0.502, 0.671], [0.030, 0.093, 0.063]),
                "ibs": metric_entry([0.140, 0.682, 0.132], [0.004, 0.099, 0.005]),
            },
            200: {
                "c_index": metric_entry([0.838, 0.570, 0.700], [0.017, 0.090, 0.037]),
                "auc": metric_entry([0.768, 0.427, 0.555], [0.030, 0.094, 0.085]),
                "ibs": metric_entry([0.157, 0.201, 0.187], [0.009, 0.028, 0.013]),
            },
        },
    },
}


def aggregate_metric(metric_data: dict[str, list[float]]) -> tuple[float, float]:
    means = metric_data["means"]
    stds = metric_data["stds"]
    mean_value = float(np.mean(means))
    propagated_std = math.sqrt(sum(std * std for std in stds)) / len(stds)
    return mean_value, propagated_std


def reduce_paradigm_results(client_count: int, paradigm: str, metric_name: str) -> tuple[float, float, int]:
    fixed_tree_config = {
        "Local": FIXED_LOCAL_TREE,
        "Federated": FIXED_FED_TREE,
        "Centralized": FIXED_CENTRALIZED_TREE,
    }[paradigm]
    mean_value, std_value = aggregate_metric(RAW_RESULTS[client_count][paradigm][fixed_tree_config][metric_name])
    return mean_value, std_value, fixed_tree_config


def get_plot_values(metric_name: str) -> tuple[dict[int, list[float]], dict[int, list[float]], dict[int, dict[str, int]]]:
    values_by_config: dict[int, list[float]] = {}
    errors_by_config: dict[int, list[float]] = {}
    selected_tree_by_config: dict[int, dict[str, int]] = {}

    for client_count in CLIENT_COUNTS:
        values = []
        errors = []
        selected_trees = {}

        for paradigm in PARADIGMS:
            mean_value, std_value, tree_config = reduce_paradigm_results(client_count, paradigm, metric_name)
            values.append(mean_value)
            errors.append(std_value)
            selected_trees[paradigm] = tree_config

        values_by_config[client_count] = values
        errors_by_config[client_count] = errors
        selected_tree_by_config[client_count] = selected_trees

    return values_by_config, errors_by_config, selected_tree_by_config


def add_value_labels(ax: plt.Axes, bars, values: list[float], errors: list[float]) -> None:
    y_min, y_max = ax.get_ylim()
    label_offset = (y_max - y_min) * 0.03
    for bar, value, error in zip(bars, values, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + label_offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
        )


def plot_e2_main_result() -> None:
    metric_specs = [
        ("c_index", "C-index"),
        ("auc", "AUC"),
        ("ibs", "IBS"),
    ]
    client_counts_for_plot = [3, 4, 5]
    x = np.arange(len(client_counts_for_plot))
    width = 0.18
    offsets = {
        "Local": -width,
        "Federated": 0.0,
        "Centralized": width,
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    reduction_notes = []
    mean_summaries = []

    for ax, (metric_key, title) in zip(axes, metric_specs):
        values_by_config, errors_by_config, selected_tree_by_config = get_plot_values(metric_key)

        all_values = []
        all_errors = []

        for paradigm in PARADIGMS:
            values = []
            errors = []
            for client_count in client_counts_for_plot:
                paradigm_index = PARADIGMS.index(paradigm)
                values.append(values_by_config[client_count][paradigm_index])
                errors.append(errors_by_config[client_count][paradigm_index])

            bars = ax.bar(
                x + offsets[paradigm],
                values,
                width=width,
                yerr=errors,
                capsize=4,
                color=COLORS[paradigm],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9,
            )
            all_values.extend(values)
            all_errors.extend(errors)
            add_value_labels(ax, bars, values, errors)

        ax.set_xticks(x)
        ax.set_xticklabels([str(client_count) for client_count in client_counts_for_plot], fontsize=TICK_FONT_SIZE)
        ax.set_xlabel("Number of clients", fontsize=LABEL_FONT_SIZE)
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold")
        ax.set_ylabel("Performance", fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        upper_limit = max(value + error for value, error in zip(all_values, all_errors))
        ax.set_ylim(0, upper_limit + 0.18)

        note = ", ".join(
            f"{client_count}C({paradigm[:3]})={selected_tree_by_config[client_count][paradigm]}"
            for client_count in client_counts_for_plot
            for paradigm in PARADIGMS
        )
        reduction_notes.append(f"{title}: {note}")
        mean_summaries.append(
            (
                title,
                {
                    paradigm: [
                        values_by_config[client_count][PARADIGMS.index(paradigm)]
                        for client_count in client_counts_for_plot
                    ]
                    for paradigm in PARADIGMS
                },
            )
        )

    legend_handles = [
        Patch(facecolor=COLORS[paradigm], edgecolor="black", label=paradigm)
        for paradigm in PARADIGMS
    ]
    fig.legend(
        legend_handles,
        PARADIGMS,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.12, top=0.84, wspace=0.28)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

    print(f"Saved figure to: {OUTPUT_PATH}")
    print("Mean values used in the plot:")
    for title, series in mean_summaries:
        print(f"{title}:")
        for paradigm, y_values in series.items():
            values_text = ", ".join(
                f"{client_count}C={value:.3f}"
                for client_count, value in zip(client_counts_for_plot, y_values)
            )
            print(f"  - {paradigm}: {values_text}")
    print("Reduction summary:")
    for note in reduction_notes:
        print(f"  - {note}")


if __name__ == "__main__":
    plot_e2_main_result()
