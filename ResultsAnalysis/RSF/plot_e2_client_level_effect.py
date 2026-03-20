"""
RSF E2 per-client analysis for the 5-client setting.

Design:
- X-axis: clients (C0-C4)
- Y-axis: delta = Federated - Local
- Subplots: C-index, AUC, IBS

For the federated side, this script uses the best-performing federated tree
setting per metric in the 5-client configuration.
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

OUTPUT_PATH = Path(__file__).with_name("figure_3_E2_RSF.png")
CLIENT_COUNT = 5
TREE_CONFIGS = [50, 100, 200]
FIXED_LOCAL_TREE = 100
FIXED_FED_TREE = 100

METRIC_SPECS = [
    ("c_index", "C-index"),
    ("auc", "AUC"),
    ("ibs", "IBS"),
]

LOCAL_COLOR = "#D55E00"  # reddish-orange
FEDERATED_COLOR = "#0072B2"  # blue

POSITIVE_COLOR = FEDERATED_COLOR
NEGATIVE_COLOR = LOCAL_COLOR


def metric_entry(means: list[float], stds: list[float]) -> dict[str, list[float]]:
    return {"means": means, "stds": stds}


RAW_RESULTS = {
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
}


def aggregate_metric(metric_data: dict[str, list[float]]) -> float:
    return float(np.mean(metric_data["means"]))


def get_best_tree_config(paradigm: str, metric_name: str) -> int:
    scores = {
        tree_config: aggregate_metric(RAW_RESULTS[paradigm][tree_config][metric_name])
        for tree_config in TREE_CONFIGS
    }
    if metric_name == "ibs":
        return min(scores, key=scores.get)
    return max(scores, key=scores.get)


def compute_delta(metric_name: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    best_fed_tree = FIXED_FED_TREE
    best_local_tree = FIXED_LOCAL_TREE

    fed_data = RAW_RESULTS["Federated"][best_fed_tree][metric_name]
    local_data = RAW_RESULTS["Local"][best_local_tree][metric_name]

    fed_means = np.array(fed_data["means"], dtype=float)
    local_means = np.array(local_data["means"], dtype=float)
    fed_stds = np.array(fed_data["stds"], dtype=float)
    local_stds = np.array(local_data["stds"], dtype=float)

    deltas = fed_means - local_means
    delta_stds = np.sqrt(fed_stds**2 + local_stds**2)
    return deltas, delta_stds, best_fed_tree, best_local_tree


def add_value_labels(ax: plt.Axes, bars, values: np.ndarray) -> None:
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.035
    for bar, value in zip(bars, values):
        y_text = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_text,
            f"{value:+.2f}",
            ha="center",
            va=va,
            fontsize=8,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 0.2,
            },
        )


def plot_e2_client_level_effect() -> None:
    fig, axes = plt.subplots(1, len(METRIC_SPECS), figsize=(15, 4.5), squeeze=False)
    axes = axes[0]
    reduction_notes = []

    clients = [f"C{i}" for i in range(CLIENT_COUNT)]

    for ax, (metric_key, metric_label) in zip(axes, METRIC_SPECS):
        deltas, delta_stds, fed_tree, local_tree = compute_delta(metric_key)
        x = np.arange(CLIENT_COUNT)
        colors = [POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR for value in deltas]

        bars = ax.bar(
            x,
            deltas,
            yerr=delta_stds,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )

        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(clients, fontsize=11)
        ax.set_xlabel("Clients", fontsize=11)
        ax.set_ylabel("Delta (Federated - Local)", fontsize=11)
        ax.set_title(metric_label, fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        limit = max(np.max(np.abs(deltas + delta_stds)), np.max(np.abs(deltas - delta_stds)))
        ax.set_ylim(-(limit + 0.12), limit + 0.12)
        add_value_labels(ax, bars, deltas)

        reduction_notes.append(f"{metric_label}: Fed={fed_tree}, Local={local_tree}")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")
    print("Reduction summary:")
    for note in reduction_notes:
        print(f"  - {note}")


if __name__ == "__main__":
    plot_e2_client_level_effect()
