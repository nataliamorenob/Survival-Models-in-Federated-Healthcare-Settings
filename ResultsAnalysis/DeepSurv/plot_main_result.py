"""
Figure 1 main result across training paradigms.

Design:
- X-axis: Local | Federated | Centralized
- Grouped bars: 5 clients, 4 clients, 3 clients
- Subplots: C-index, AUC, IBS

The federated setting contains three strategies in the attached results
(FedAvg, FedProx, FedAdam). Since this figure compares training paradigms
rather than FL strategies, the script summarizes federated performance with
the FedAvg strategy for each metric and client setting.
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

OUTPUT_PATH = Path(__file__).with_name("figure_1_DeepSurv.png")
FEDERATED_STRATEGY = "FedAvg"

PARADIGMS = ["Local", "Federated", "Centralized"]
CLIENT_CONFIGS = [5, 4, 3]
LOCAL_COLOR = "#D55E00"  # reddish-orange
FEDERATED_COLOR = "#0072B2"  # blue
CENTRALIZED_COLOR = "#009E73"  # bluish-green

colors = {
    "Local": LOCAL_COLOR,
    "Federated": FEDERATED_COLOR,
    "Centralized": CENTRALIZED_COLOR,
}


def metric_entry(means: list[float], stds: list[float]) -> dict[str, list[float]]:
    return {"means": means, "stds": stds}


RAW_RESULTS = {
    5: {
        "Local": {
            "c_index": metric_entry([0.740, 0.521, 0.543, 0.509, 0.775], [0.130, 0.166, 0.118, 0.119, 0.161]),
            "auc": metric_entry([0.689, 0.515, 0.582, 0.510, 0.807], [0.102, 0.163, 0.120, 0.124, 0.182]),
            "ibs": metric_entry([0.179, 0.270, 0.195, 0.103, 0.046], [0.015, 0.047, 0.011, 0.007, 0.005]),
        },
        "Centralized": {
            "c_index": metric_entry([0.773, 0.723, 0.616, 0.570, 0.851], [0.145, 0.149, 0.147, 0.155, 0.188]),
            "auc": metric_entry([0.737, 0.724, 0.541, 0.550, 0.869], [0.129, 0.162, 0.202, 0.195, 0.198]),
            "ibs": metric_entry([0.167, 0.841, 0.158, 0.141, 0.040], [0.019, 0.225, 0.016, 0.010, 0.004]),
        },
        "Federated": {
            "FedAvg": {
                "c_index": metric_entry([0.804, 0.640, 0.712, 0.551, 0.929], [0.055, 0.192, 0.090, 0.183, 0.063]),
                "auc": metric_entry([0.763, 0.639, 0.606, 0.569, 0.934], [0.054, 0.205, 0.124, 0.208, 0.112]),
                "ibs": metric_entry([0.162, 0.243, 0.189, 0.101, 0.045], [0.011, 0.030, 0.015, 0.007, 0.004]),
            },
            "FedProx": {
                "c_index": metric_entry([0.798, 0.644, 0.710, 0.533, 0.910], [0.056, 0.176, 0.095, 0.131, 0.095]),
                "auc": metric_entry([0.759, 0.647, 0.603, 0.555, 0.922], [0.055, 0.182, 0.146, 0.168, 0.138]),
                "ibs": metric_entry([0.161, 0.241, 0.190, 0.101, 0.046], [0.010, 0.031, 0.015, 0.008, 0.004]),
            },
            "FedAdam": {
                "c_index": metric_entry([0.744, 0.630, 0.676, 0.519, 0.868], [0.133, 0.208, 0.133, 0.134, 0.153]),
                "auc": metric_entry([0.712, 0.608, 0.582, 0.530, 0.895], [0.117, 0.212, 0.122, 0.169, 0.181]),
                "ibs": metric_entry([0.165, 0.248, 0.189, 0.101, 0.045], [0.010, 0.032, 0.017, 0.008, 0.004]),
            },
        },
    },
    4: {
        "Local": {
            "c_index": metric_entry([0.740, 0.521, 0.543, 0.509], [0.130, 0.166, 0.118, 0.119]),
            "auc": metric_entry([0.689, 0.515, 0.582, 0.510], [0.102, 0.163, 0.120, 0.124]),
            "ibs": metric_entry([0.179, 0.270, 0.195, 0.103], [0.015, 0.047, 0.011, 0.007]),
        },
        "Centralized": {
            "c_index": metric_entry([0.756, 0.686, 0.694, 0.571], [0.136, 0.179, 0.115, 0.187]),
            "auc": metric_entry([0.712, 0.683, 0.636, 0.571], [0.116, 0.169, 0.121, 0.214]),
            "ibs": metric_entry([0.152, 0.744, 0.138, 0.116], [0.013, 0.083, 0.009, 0.012]),
        },
        "Federated": {
            "FedAvg": {
                "c_index": metric_entry([0.803, 0.615, 0.728, 0.539], [0.025, 0.158, 0.085, 0.147]),
                "auc": metric_entry([0.753, 0.604, 0.627, 0.561], [0.021, 0.170, 0.131, 0.164]),
                "ibs": metric_entry([0.164, 0.246, 0.185, 0.098], [0.013, 0.031, 0.013, 0.008]),
            },
            "FedProx": {
                "c_index": metric_entry([0.790, 0.613, 0.729, 0.550], [0.036, 0.158, 0.086, 0.178]),
                "auc": metric_entry([0.741, 0.607, 0.656, 0.574], [0.032, 0.171, 0.126, 0.203]),
                "ibs": metric_entry([0.163, 0.244, 0.186, 0.099], [0.012, 0.027, 0.014, 0.007]),
            },
            "FedAdam": {
                "c_index": metric_entry([0.739, 0.618, 0.687, 0.555], [0.123, 0.165, 0.141, 0.143]),
                "auc": metric_entry([0.699, 0.602, 0.607, 0.575], [0.101, 0.187, 0.143, 0.148]),
                "ibs": metric_entry([0.167, 0.248, 0.183, 0.098], [0.013, 0.033, 0.015, 0.009]),
            },
        },
    },
    3: {
        "Local": {
            "c_index": metric_entry([0.740, 0.521, 0.543], [0.130, 0.166, 0.118]),
            "auc": metric_entry([0.689, 0.515, 0.582], [0.102, 0.163, 0.120]),
            "ibs": metric_entry([0.179, 0.270, 0.195], [0.015, 0.047, 0.011]),
        },
        "Centralized": {
            "c_index": metric_entry([0.717, 0.584, 0.649], [0.154, 0.132, 0.106]),
            "auc": metric_entry([0.677, 0.594, 0.578], [0.131, 0.117, 0.102]),
            "ibs": metric_entry([0.163, 0.730, 0.148], [0.0123, 0.084, 0.016]),
        },
        "Federated": {
            "FedAvg": {
                "c_index": metric_entry([0.808, 0.610, 0.754], [0.025, 0.166, 0.063]),
                "auc": metric_entry([0.751, 0.592, 0.657], [0.024, 0.194, 0.117]),
                "ibs": metric_entry([0.165, 0.249, 0.180], [0.013, 0.040, 0.015]),
            },
            "FedProx": {
                "c_index": metric_entry([0.798, 0.620, 0.745], [0.025, 0.148, 0.068]),
                "auc": metric_entry([0.741, 0.609, 0.678], [0.024, 0.175, 0.126]),
                "ibs": metric_entry([0.165, 0.244, 0.182], [0.012, 0.035, 0.016]),
            },
            "FedAdam": {
                "c_index": metric_entry([0.745, 0.618, 0.709], [0.120, 0.170, 0.119]),
                "auc": metric_entry([0.705, 0.611, 0.636], [0.099, 0.183, 0.135]),
                "ibs": metric_entry([0.168, 0.251, 0.182], [0.011, 0.040, 0.017]),
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


def reduce_federated_results(client_count: int, metric_name: str) -> tuple[float, float, str]:
    metrics = RAW_RESULTS[client_count]["Federated"][FEDERATED_STRATEGY]
    mean_value, propagated_std = aggregate_metric(metrics[metric_name])
    return mean_value, propagated_std, FEDERATED_STRATEGY


def get_plot_values(metric_name: str) -> tuple[dict[int, list[float]], dict[int, list[float]], dict[int, str]]:
    values_by_config: dict[int, list[float]] = {}
    errors_by_config: dict[int, list[float]] = {}
    federated_strategy_by_config: dict[int, str] = {}

    for client_count in CLIENT_CONFIGS:
        local_value, local_error = aggregate_metric(RAW_RESULTS[client_count]["Local"][metric_name])
        federated_value, federated_error, federated_strategy = reduce_federated_results(client_count, metric_name)
        centralized_value, centralized_error = aggregate_metric(RAW_RESULTS[client_count]["Centralized"][metric_name])

        values_by_config[client_count] = [local_value, federated_value, centralized_value]
        errors_by_config[client_count] = [local_error, federated_error, centralized_error]
        federated_strategy_by_config[client_count] = federated_strategy

    return values_by_config, errors_by_config, federated_strategy_by_config


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
            fontsize=9.6,
        )


def plot_main_result() -> None:
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

    federated_notes = []

    for ax, (metric_key, title) in zip(axes, metric_specs):
        values_by_config, errors_by_config, federated_strategy_by_config = get_plot_values(metric_key)

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
                color=colors[paradigm],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9,
            )
            all_values.extend(values)
            all_errors.extend(errors)
            add_value_labels(ax, bars, values, errors)

        ax.set_xticks(x)
        ax.set_xticklabels([str(client_count) for client_count in client_counts_for_plot], fontsize=12.6)
        ax.set_xlabel("Number of clients", fontsize=12.6)
        ax.set_title(title, fontsize=14.6, fontweight="bold")
        ax.set_ylabel("Performance", fontsize=12.6)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        upper_limit = max(value + error for value, error in zip(all_values, all_errors))
        ax.set_ylim(0, upper_limit + 0.18)

        strategy_note = ", ".join(
            f"{client_count}C={strategy}"
            for client_count, strategy in federated_strategy_by_config.items()
        )
        federated_notes.append(f"{title}: {strategy_note}")

    legend_handles = [
        Patch(facecolor=colors[paradigm], edgecolor="black", label=paradigm)
        for paradigm in PARADIGMS
    ]
    fig.legend(legend_handles, PARADIGMS, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.12, top=0.84, wspace=0.28)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

    print(f"Saved figure to: {OUTPUT_PATH}")
    print("Federated reduction summary:")
    for note in federated_notes:
        print(f"  - {note}")


if __name__ == "__main__":
    plot_main_result()
