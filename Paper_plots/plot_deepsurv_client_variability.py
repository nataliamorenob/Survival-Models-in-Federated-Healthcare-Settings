"""
Create a DeepSurv client-level variability figure for the paper.

Figure layout:
- row: client configuration (5 clients)
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

OUTPUT_PATH = Path(__file__).with_name("figure_2_deepsurv_client_variability.png")

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
            "c_index": metric_entry([0.740, 0.521, 0.543, 0.509, 0.775], [0.130, 0.166, 0.118, 0.119, 0.161]),
            "auc": metric_entry([0.689, 0.515, 0.582, 0.510, 0.807], [0.102, 0.163, 0.120, 0.124, 0.182]),
            "ibs": metric_entry([0.179, 0.270, 0.195, 0.103, 0.046], [0.015, 0.047, 0.011, 0.007, 0.005]),
        },
        "Federated": {
            "c_index": metric_entry([0.804, 0.640, 0.712, 0.551, 0.929], [0.055, 0.192, 0.090, 0.183, 0.063]),
            "auc": metric_entry([0.763, 0.639, 0.606, 0.569, 0.934], [0.054, 0.205, 0.124, 0.208, 0.112]),
            "ibs": metric_entry([0.162, 0.243, 0.189, 0.101, 0.045], [0.011, 0.030, 0.015, 0.007, 0.004]),
        },
        "Centralized": {
            "c_index": metric_entry([0.773, 0.723, 0.616, 0.570, 0.851], [0.145, 0.149, 0.147, 0.155, 0.188]),
            "auc": metric_entry([0.737, 0.724, 0.541, 0.550, 0.869], [0.129, 0.162, 0.202, 0.195, 0.198]),
            "ibs": metric_entry([0.167, 0.041, 0.158, 0.141, 0.040], [0.019, 0.225, 0.016, 0.010, 0.004]),
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
        "Client-level performance variability for DeepSurv",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_client_variability()
