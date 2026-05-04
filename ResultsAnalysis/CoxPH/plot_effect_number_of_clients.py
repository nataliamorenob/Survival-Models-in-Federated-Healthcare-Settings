"""
Figure 3: effect of number of clients.

Design:
- X-axis: number of clients (3, 4, 5)
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

OUTPUT_PATH = Path(__file__).with_name("figure_3_effect_number_of_clients.png")
CLIENT_COUNTS = [3, 4, 5]
FEDERATED_REDUCTION = "best_per_metric"

LOCAL_COLOR = "#D55E00"  # reddish-orange
FEDERATED_COLOR = "#0072B2"  # blue
CENTRALIZED_COLOR = "#009E73"  # bluish-green

LINE_STYLES = {
    "Local": {"color": LOCAL_COLOR, "marker": "o"},
    "Federated": {"color": FEDERATED_COLOR, "marker": "s"},
    "Centralized": {"color": CENTRALIZED_COLOR, "marker": "^"},
}

TITLE_FONT_SIZE = 17
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 11.5
LEGEND_FONT_SIZE = 13

METRIC_SPECS = [
    ("c_index", "C-index"),
    ("auc", "AUC"),
    ("ibs", "IBS"),
]


RESULTS = {
    "Local": {
        5: {"c_index": 0.598, "auc": 0.606, "ibs": 0.167},
        4: {"c_index": 0.566, "auc": 0.564, "ibs": 0.197},
        3: {"c_index": 0.570, "auc": 0.554, "ibs": 0.227},
    },
    "Centralized": {
        5: {"c_index": 0.732, "auc": 0.697, "ibs": 0.290},
        4: {"c_index": 0.702, "auc": 0.659, "ibs": 0.288},
        3: {"c_index": 0.699, "auc": 0.658, "ibs": 0.368},
    },
    "Federated": {
        5: {
            "FedAvg": {"c_index": 0.611, "auc": 0.600, "ibs": 0.155},
            "FedProx": {"c_index": 0.607, "auc": 0.592, "ibs": 0.156},
            "FedAdam": {"c_index": 0.573, "auc": 0.553, "ibs": 0.160},
        },
        4: {
            "FedAvg": {"c_index": 0.594, "auc": 0.581, "ibs": 0.183},
            "FedProx": {"c_index": 0.600, "auc": 0.583, "ibs": 0.181},
            "FedAdam": {"c_index": 0.561, "auc": 0.549, "ibs": 0.188},
        },
        3: {
            "FedAvg": {"c_index": 0.609, "auc": 0.573, "ibs": 0.213},
            "FedProx": {"c_index": 0.605, "auc": 0.574, "ibs": 0.213},
            "FedAdam": {"c_index": 0.568, "auc": 0.543, "ibs": 0.217},
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


def add_value_labels(ax: plt.Axes, x_values: list[int], series: dict[str, list[float]]) -> None:
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    close_threshold = 0.06 * y_range
    base_offsets = {
        "Local": (-10, 8),
        "Federated": (0, 10),
        "Centralized": (10, 8),
    }

    for index, x_value in enumerate(x_values):
        points = [
            {
                "paradigm": paradigm,
                "y": y_values[index],
                "color": LINE_STYLES[paradigm]["color"],
            }
            for paradigm, y_values in series.items()
        ]
        label_offsets = {
            point["paradigm"]: list(base_offsets[point["paradigm"]])
            for point in points
        }

        if index == 0:
            for offsets in label_offsets.values():
                offsets[0] = abs(offsets[0]) + 4
        elif index == len(x_values) - 1:
            for offsets in label_offsets.values():
                offsets[0] = -abs(offsets[0]) - 4

        points_by_height = sorted(points, key=lambda point: point["y"])
        lower_gap = points_by_height[1]["y"] - points_by_height[0]["y"]
        upper_gap = points_by_height[2]["y"] - points_by_height[1]["y"]

        if lower_gap < close_threshold and upper_gap < close_threshold:
            label_offsets[points_by_height[0]["paradigm"]][1] = -14
            label_offsets[points_by_height[1]["paradigm"]][1] = 0
            label_offsets[points_by_height[2]["paradigm"]][1] = 14
        elif lower_gap < close_threshold:
            label_offsets[points_by_height[0]["paradigm"]][1] = -12
            label_offsets[points_by_height[1]["paradigm"]][1] = 12
        elif upper_gap < close_threshold:
            label_offsets[points_by_height[1]["paradigm"]][1] = -12
            label_offsets[points_by_height[2]["paradigm"]][1] = 12

        for point in points:
            dx, dy = label_offsets[point["paradigm"]]
            ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
            va = "bottom" if dy >= 0 else "top"
            ax.annotate(
                f"{point['y']:.2f}",
                xy=(x_value, point["y"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=ANNOTATION_FONT_SIZE,
                color=point["color"],
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
    mean_summaries = []

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
        mean_summaries.append((metric_label, series))

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

        ax.set_title(metric_label, fontsize=TITLE_FONT_SIZE, fontweight="bold")
        ax.set_xlabel("Number of clients", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Mean performance", fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(CLIENT_COUNTS)
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        ax.set_xlim(2.85, 5.15)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        metric_values = [value for values in series.values() for value in values]
        padding = 0.06 if metric_key != "ibs" else 0.04
        ax.set_ylim(min(metric_values) - padding, max(metric_values) + padding)
        add_value_labels(ax, CLIENT_COUNTS, series)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

    print(f"Saved figure to: {OUTPUT_PATH}")
    print("Mean values used in the plot:")
    for metric_label, series in mean_summaries:
        print(f"{metric_label}:")
        for paradigm, y_values in series.items():
            values_text = ", ".join(
                f"{client_count}C={value:.3f}"
                for client_count, value in zip(CLIENT_COUNTS, y_values)
            )
            print(f"  - {paradigm}: {values_text}")
    print("Federated reduction summary:")
    for note in federated_notes:
        print(f"  - {note}")


if __name__ == "__main__":
    plot_effect_number_of_clients()
