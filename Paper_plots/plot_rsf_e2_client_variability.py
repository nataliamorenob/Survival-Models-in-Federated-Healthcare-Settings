"""
Create an RSF E2 client-level variability figure for the paper.

Figure layout:
- row: client configuration (5 clients, 100 trees)
- columns: C-index, AUC, IBS
- each subplot: grouped bars for Local, Federated, Centralized
- x-axis: clients available in that configuration
- error bars: standard deviation from the reported 10-run summary tables
"""

from __future__ import annotations

import os
from pathlib import Path

MPL_CONFIG_DIR = Path(__file__).with_name(".mplconfig")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


plt.style.use("seaborn-v0_8-whitegrid")

OUTPUT_PATH = Path(__file__).with_name("figure_2_rsf_e2_client_variability.png")

LOCAL_COLOR = "#D55E00"
FEDERATED_COLOR = "#0072B2"
CENTRALIZED_COLOR = "#009E73"

PARADIGMS = ["Local", "Federated", "Centralized"]
COLORS = {
    "Local": LOCAL_COLOR,
    "Federated": FEDERATED_COLOR,
    "Centralized": CENTRALIZED_COLOR,
}

CLIENT_COUNTS_TO_PLOT = [5]
METRIC_SPECS = [
    ("c_index", "C-index"),
    ("auc", "AUC"),
    ("ibs", "IBS"),
]

TITLE_FONT_SIZE = 15
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 11
LEGEND_FONT_SIZE = 12
ROW_LABEL_FONT_SIZE = 13


def metric_entry(means: list[float], stds: list[float]) -> dict[str, list[float]]:
    return {"means": means, "stds": stds}


RESULTS = {
    5: {
        "Local": {
            "c_index": metric_entry([0.824, 0.368, 0.658, 0.608, 0.978], [0.015, 0.098, 0.090, 0.103, 0.031]),
            "auc": metric_entry([0.745, 0.374, 0.643, 0.613, 0.951], [0.036, 0.118, 0.102, 0.104, 0.021]),
            "ibs": metric_entry([0.150, 0.283, 0.189, 0.096, 0.043], [0.010, 0.047, 0.011, 0.010, 0.001]),
        },
        "Federated": {
            "c_index": metric_entry([0.821, 0.520, 0.684, 0.658, 0.938], [0.020, 0.130, 0.057, 0.070, 0.051]),
            "auc": metric_entry([0.765, 0.443, 0.512, 0.731, 0.937], [0.026, 0.146, 0.092, 0.046, 0.032]),
            "ibs": metric_entry([0.163, 0.191, 0.191, 0.096, 0.047], [0.007, 0.018, 0.010, 0.006, 0.001]),
        },
        "Centralized": {
            "c_index": metric_entry([0.823, 0.600, 0.744, 0.670, 0.895], [0.013, 0.108, 0.044, 0.071, 0.066]),
            "auc": metric_entry([0.765, 0.578, 0.623, 0.749, 0.885], [0.030, 0.087, 0.075, 0.065, 0.033]),
            "ibs": metric_entry([0.154, 0.766, 0.143, 0.124, 0.039], [0.006, 0.119, 0.006, 0.009, 0.001]),
        },
    }
}


def compute_metric_limits(metric_name: str) -> tuple[float, float]:
    min_value = float("inf")
    max_value = float("-inf")

    for client_count in CLIENT_COUNTS_TO_PLOT:
        for paradigm in PARADIGMS:
            metric_data = RESULTS[client_count][paradigm][metric_name]
            means = np.array(metric_data["means"], dtype=float)
            stds = np.array(metric_data["stds"], dtype=float)
            min_value = min(min_value, float(np.min(means - stds)))
            max_value = max(max_value, float(np.max(means + stds)))

    padding = 0.04 if metric_name != "ibs" else 0.05
    return max(0.0, min_value - padding), min(1.2, max_value + padding)


def plot_client_variability() -> None:
    fig, axes = plt.subplots(
        len(CLIENT_COUNTS_TO_PLOT),
        len(METRIC_SPECS),
        figsize=(16, 4.6),
        sharey="col",
        squeeze=False,
    )
    width = 0.24
    offsets = np.array([-width, 0.0, width])
    metric_limits = {
        metric_key: compute_metric_limits(metric_key)
        for metric_key, _ in METRIC_SPECS
    }

    for row_index, client_count in enumerate(CLIENT_COUNTS_TO_PLOT):
        clients = [f"C{i}" for i in range(client_count)]
        x = np.arange(client_count, dtype=float)

        for col_index, (metric_key, metric_label) in enumerate(METRIC_SPECS):
            ax = axes[row_index, col_index]

            for offset, paradigm in zip(offsets, PARADIGMS):
                metric_data = RESULTS[client_count][paradigm][metric_key]
                means = np.array(metric_data["means"], dtype=float)
                stds = np.array(metric_data["stds"], dtype=float)

                ax.bar(
                    x + offset,
                    means,
                    width=width,
                    yerr=stds,
                    capsize=3,
                    color=COLORS[paradigm],
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=3,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(clients, fontsize=TICK_FONT_SIZE)
            ax.set_ylim(*metric_limits[metric_key])
            ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

            if row_index == 0:
                ax.set_title(metric_label, fontsize=TITLE_FONT_SIZE, fontweight="bold")

            if col_index == 0:
                ax.set_ylabel(
                    f"{client_count} clients\nPerformance",
                    fontsize=ROW_LABEL_FONT_SIZE,
                    fontweight="bold",
                )

            if row_index == len(CLIENT_COUNTS_TO_PLOT) - 1:
                ax.set_xlabel("Client", fontsize=LABEL_FONT_SIZE)

    legend_handles = [
        Patch(facecolor=COLORS[paradigm], edgecolor="black", label=paradigm)
        for paradigm in PARADIGMS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.suptitle(
        "Client-level performance variability for RSF E2",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_client_variability()
