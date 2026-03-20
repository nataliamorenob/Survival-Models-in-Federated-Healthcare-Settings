"""
Figure 3: effect of number of clients.

Design:
- X-axis: number of clients (5, 4, 3)
- Y-axis: mean performance
- Type: line plot
- Lines: Local, Federated, Centralized

The attached table reports federated results per strategy. To collapse them into
one "Federated" line, this script defaults to the best-performing federated
strategy per metric and client count, consistent with Figure 1. If you prefer a
single line averaged across FedAvg, FedProx, and FedAdam, change
FEDERATED_REDUCTION to "mean_across_strategies".
"""

from __future__ import annotations

import os
from pathlib import Path

MPL_CONFIG_DIR = Path(__file__).with_name(".mplconfig")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np


plt.style.use("seaborn-v0_8-whitegrid")

OUTPUT_PATH = Path(__file__).with_name("figure_3_DeepSurv.png")
CLIENT_COUNTS = [5, 4, 3]
FEDERATED_REDUCTION = "best_per_metric"

LOCAL_COLOR = "#D55E00"  # reddish-orange
FEDERATED_COLOR = "#0072B2"  # blue
CENTRALIZED_COLOR = "#009E73"  # bluish-green

LINE_STYLES = {
    "Local": {"color": LOCAL_COLOR, "marker": "o"},
    "Federated": {"color": FEDERATED_COLOR, "marker": "s"},
    "Centralized": {"color": CENTRALIZED_COLOR, "marker": "^"},
}

METRIC_SPECS = [
    ("c_index", "C-index"),
    ("auc", "AUC"),
    ("ibs", "IBS"),
]


RESULTS = {
    "Local": {
        5: {"c_index": 0.618, "auc": 0.621, "ibs": 0.159},
        4: {"c_index": 0.578, "auc": 0.574, "ibs": 0.187},
        3: {"c_index": 0.601, "auc": 0.595, "ibs": 0.215},
    },
    "Centralized": {
        5: {"c_index": 0.707, "auc": 0.684, "ibs": 0.269},
        4: {"c_index": 0.677, "auc": 0.651, "ibs": 0.288},
        3: {"c_index": 0.650, "auc": 0.616, "ibs": 0.347},
    },
    "Federated": {
        5: {
            "FedAvg": {"c_index": 0.727, "auc": 0.702, "ibs": 0.148},
            "FedProx": {"c_index": 0.719, "auc": 0.697, "ibs": 0.148},
            "FedAdam": {"c_index": 0.687, "auc": 0.665, "ibs": 0.150},
        },
        4: {
            "FedAvg": {"c_index": 0.671, "auc": 0.636, "ibs": 0.173},
            "FedProx": {"c_index": 0.671, "auc": 0.644, "ibs": 0.173},
            "FedAdam": {"c_index": 0.650, "auc": 0.621, "ibs": 0.174},
        },
        3: {
            "FedAvg": {"c_index": 0.724, "auc": 0.667, "ibs": 0.198},
            "FedProx": {"c_index": 0.721, "auc": 0.676, "ibs": 0.197},
            "FedAdam": {"c_index": 0.691, "auc": 0.651, "ibs": 0.200},
        },
    },
}


def reduce_federated(metric_name: str) -> tuple[list[float], list[str]]:
    values = []
    selected_strategies = []

    for client_count in CLIENT_COUNTS:
        strategy_values = RESULTS["Federated"][client_count]

        if FEDERATED_REDUCTION == "mean_across_strategies":
            reduced_value = float(np.mean([metrics[metric_name] for metrics in strategy_values.values()]))
            values.append(reduced_value)
            selected_strategies.append("Mean FL")
            continue

        if metric_name == "ibs":
            selected_strategy = min(strategy_values, key=lambda strategy: strategy_values[strategy][metric_name])
        else:
            selected_strategy = max(strategy_values, key=lambda strategy: strategy_values[strategy][metric_name])

        values.append(strategy_values[selected_strategy][metric_name])
        selected_strategies.append(selected_strategy)

    return values, selected_strategies


def add_value_labels(ax: plt.Axes, x_values: list[int], y_values: list[float], color: str) -> None:
    offset_map = {
        LOCAL_COLOR: (-10, 8),
        FEDERATED_COLOR: (0, 10),
        CENTRALIZED_COLOR: (10, 8),
    }
    dx, dy = offset_map[color]
    for x_value, y_value in zip(x_values, y_values):
        ax.annotate(
            f"{y_value:.2f}",
            xy=(x_value, y_value),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 0.2,
            },
        )


def plot_effect_number_of_clients() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    federated_notes = []

    for ax, (metric_key, metric_label) in zip(axes, METRIC_SPECS):
        federated_values, selected_strategies = reduce_federated(metric_key)
        federated_notes.append(
            f"{metric_label}: " + ", ".join(
                f"{client_count}C={strategy}"
                for client_count, strategy in zip(CLIENT_COUNTS, selected_strategies)
            )
        )

        series = {
            "Local": [RESULTS["Local"][client_count][metric_key] for client_count in CLIENT_COUNTS],
            "Federated": federated_values,
            "Centralized": [RESULTS["Centralized"][client_count][metric_key] for client_count in CLIENT_COUNTS],
        }

        for paradigm, y_values in series.items():
            style = LINE_STYLES[paradigm]
            ax.plot(
                CLIENT_COUNTS,
                y_values,
                linewidth=2.2,
                markersize=7,
                label=paradigm,
                **style,
            )
            add_value_labels(ax, CLIENT_COUNTS, y_values, style["color"])

        ax.set_title(metric_label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of clients", fontsize=11)
        ax.set_ylabel("Mean performance", fontsize=11)
        ax.set_xticks(CLIENT_COUNTS)
        ax.set_xlim(5.15, 2.85)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        metric_values = [value for values in series.values() for value in values]
        padding = 0.06 if metric_key != "ibs" else 0.04
        ax.set_ylim(min(metric_values) - padding, max(metric_values) + padding)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

    print(f"Saved figure to: {OUTPUT_PATH}")
    print("Federated reduction summary:")
    for note in federated_notes:
        print(f"  - {note}")


if __name__ == "__main__":
    plot_effect_number_of_clients()
