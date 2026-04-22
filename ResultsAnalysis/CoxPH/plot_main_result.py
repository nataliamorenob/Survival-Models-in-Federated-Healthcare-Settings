"""
Figure 1 main result across training paradigms.

Design:
- X-axis: Local | Federated | Centralized
- Grouped bars: 5 clients, 4 clients, 3 clients
- Subplots: C-index, AUC, IBS

The federated setting contains three strategies in the attached results
(FedAvg, FedProx, FedAdam). Since this figure compares training paradigms
rather than FL strategies, the script summarizes federated performance with
the best-performing federated strategy for each metric and client setting by
default. Change FEDERATED_REDUCTION below if you want a different reduction.
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

OUTPUT_PATH = Path(__file__).with_name("figure_1_CoxPH.png")
FEDERATED_REDUCTION = "best_per_metric"

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
            "c_index": metric_entry([0.784, 0.350, 0.577, 0.552, 0.725], [0.045, 0.156, 0.131, 0.221, 0.254]),
            "auc": metric_entry([0.718, 0.305, 0.640, 0.594, 0.771], [0.054, 0.158, 0.134, 0.248, 0.275]),
            "ibs": metric_entry([0.169, 0.314, 0.198, 0.106, 0.047], [0.018, 0.052, 0.018, 0.019, 0.004]),
        },
        "Centralized": {
            "c_index": metric_entry([0.873, 0.590, 0.666, 0.581, 0.948], [0.016, 0.131, 0.093, 0.181, 0.030]),
            "auc": metric_entry([0.832, 0.600, 0.583, 0.521, 0.951], [0.031, 0.124, 0.129, 0.211, 0.034]),
            "ibs": metric_entry([0.146, 0.965, 0.160, 0.143, 0.038], [0.008, 0.184, 0.016, 0.017, 0.005]),
        },
        "Federated": {
            "FedAvg": {
                "c_index": metric_entry([0.712, 0.481, 0.646, 0.514, 0.702], [0.139, 0.218, 0.129, 0.231, 0.234]),
                "auc": metric_entry([0.669, 0.476, 0.598, 0.554, 0.705], [0.137, 0.240, 0.154, 0.268, 0.263]),
                "ibs": metric_entry([0.169, 0.262, 0.194, 0.103, 0.049], [0.023, 0.040, 0.016, 0.013, 0.002]),
            },
            "FedProx": {
                "c_index": metric_entry([0.705, 0.472, 0.638, 0.514, 0.706], [0.144, 0.225, 0.138, 0.239, 0.226]),
                "auc": metric_entry([0.670, 0.462, 0.586, 0.545, 0.695], [0.143, 0.251, 0.157, 0.268, 0.255]),
                "ibs": metric_entry([0.170, 0.265, 0.195, 0.103, 0.049], [0.022, 0.040, 0.017, 0.014, 0.002]),
            },
            "FedAdam": {
                "c_index": metric_entry([0.646, 0.463, 0.607, 0.518, 0.631], [0.173, 0.222, 0.143, 0.233, 0.255]),
                "auc": metric_entry([0.616, 0.466, 0.558, 0.546, 0.579], [0.161, 0.247, 0.149, 0.270, 0.302]),
                "ibs": metric_entry([0.178, 0.268, 0.198, 0.105, 0.050], [0.028, 0.043, 0.017, 0.017, 0.003]),
            },
        },
    },
    4: {
        "Local": {
            "c_index": metric_entry([0.784, 0.350, 0.577, 0.552], [0.045, 0.156, 0.131, 0.221]),
            "auc": metric_entry([0.718, 0.305, 0.640, 0.594], [0.054, 0.158, 0.134, 0.248]),
            "ibs": metric_entry([0.169, 0.314, 0.198, 0.106], [0.018, 0.052, 0.018, 0.019]),
        },
        "Centralized": {
            "c_index": metric_entry([0.855, 0.631, 0.733, 0.589], [0.028, 0.105, 0.055, 0.183]),
            "auc": metric_entry([0.803, 0.616, 0.661, 0.554], [0.040, 0.105, 0.086, 0.208]),
            "ibs": metric_entry([0.135, 0.769, 0.138, 0.112], [0.011, 0.125, 0.013, 0.014]),
        },
        "Federated": {
            "FedAvg": {
                "c_index": metric_entry([0.703, 0.481, 0.666, 0.527], [0.125, 0.241, 0.123, 0.250]),
                "auc": metric_entry([0.659, 0.471, 0.635, 0.560], [0.123, 0.267, 0.142, 0.280]),
                "ibs": metric_entry([0.171, 0.266, 0.192, 0.102], [0.022, 0.042, 0.016, 0.013]),
            },
            "FedProx": {
                "c_index": metric_entry([0.701, 0.487, 0.665, 0.497], [0.142, 0.249, 0.129, 0.252]),
                "auc": metric_entry([0.660, 0.501, 0.645, 0.526], [0.139, 0.252, 0.138, 0.281]),
                "ibs": metric_entry([0.171, 0.259, 0.191, 0.104], [0.023, 0.036, 0.017, 0.014]),
            },
            "FedAdam": {
                "c_index": metric_entry([0.643, 0.452, 0.617, 0.533], [0.171, 0.227, 0.139, 0.235]),
                "auc": metric_entry([0.612, 0.455, 0.579, 0.561], [0.157, 0.248, 0.145, 0.269]),
                "ibs": metric_entry([0.179, 0.272, 0.197, 0.105], [0.028, 0.041, 0.017, 0.017]),
            },
        },
    },
    3: {
        "Local": {
            "c_index": metric_entry([0.784, 0.350, 0.577], [0.045, 0.156, 0.131]),
            "auc": metric_entry([0.718, 0.305, 0.640], [0.054, 0.158, 0.134]),
            "ibs": metric_entry([0.169, 0.314, 0.198], [0.018, 0.052, 0.018]),
        },
        "Centralized": {
            "c_index": metric_entry([0.861, 0.536, 0.700], [0.020, 0.103, 0.086]),
            "auc": metric_entry([0.807, 0.549, 0.619], [0.030, 0.073, 0.139]),
            "ibs": metric_entry([0.141, 0.810, 0.152], [0.008, 0.153, 0.020]),
        },
        "Federated": {
            "FedAvg": {
                "c_index": metric_entry([0.735, 0.445, 0.647], [0.107, 0.218, 0.126]),
                "auc": metric_entry([0.692, 0.408, 0.620], [0.101, 0.234, 0.132]),
                "ibs": metric_entry([0.168, 0.278, 0.192], [0.020, 0.041, 0.015]),
            },
            "FedProx": {
                "c_index": metric_entry([0.723, 0.443, 0.650], [0.113, 0.221, 0.128]),
                "auc": metric_entry([0.680, 0.404, 0.620], [0.105, 0.231, 0.127]),
                "ibs": metric_entry([0.169, 0.278, 0.192], [0.020, 0.040, 0.015]),
            },
            "FedAdam": {
                "c_index": metric_entry([0.641, 0.450, 0.613], [0.175, 0.196, 0.141]),
                "auc": metric_entry([0.614, 0.441, 0.574], [0.160, 0.217, 0.145]),
                "ibs": metric_entry([0.178, 0.275, 0.197], [0.028, 0.041, 0.015]),
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
    strategy_results = {}
    for strategy_name, metrics in RAW_RESULTS[client_count]["Federated"].items():
        strategy_results[strategy_name] = aggregate_metric(metrics[metric_name])

    if FEDERATED_REDUCTION == "mean_across_strategies":
        means = [result[0] for result in strategy_results.values()]
        stds = [result[1] for result in strategy_results.values()]
        mean_value = float(np.mean(means))
        propagated_std = math.sqrt(sum(std * std for std in stds)) / len(stds)
        return mean_value, propagated_std, "Mean FL"

    if metric_name == "ibs":
        selected_strategy = min(strategy_results, key=lambda name: strategy_results[name][0])
    else:
        selected_strategy = max(strategy_results, key=lambda name: strategy_results[name][0])

    mean_value, propagated_std = strategy_results[selected_strategy]
    return mean_value, propagated_std, selected_strategy


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
