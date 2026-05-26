"""
Create a CoxPH client-level variability figure for the paper.

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

OUTPUT_PATH = Path(__file__).with_name("figure_2_coxph_client_variability.png")

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
            "c_index": metric_entry([0.784, 0.350, 0.577, 0.552, 0.725], [0.045, 0.156, 0.131, 0.221, 0.254]),
            "auc": metric_entry([0.718, 0.305, 0.640, 0.594, 0.771], [0.054, 0.158, 0.134, 0.248, 0.275]),
            "ibs": metric_entry([0.169, 0.314, 0.198, 0.106, 0.047], [0.018, 0.052, 0.018, 0.019, 0.004]),
        },
        "Federated": {
            "c_index": metric_entry([0.712, 0.481, 0.646, 0.514, 0.702], [0.139, 0.218, 0.129, 0.231, 0.234]),
            "auc": metric_entry([0.669, 0.476, 0.598, 0.554, 0.705], [0.137, 0.240, 0.154, 0.268, 0.263]),
            "ibs": metric_entry([0.169, 0.262, 0.194, 0.103, 0.049], [0.023, 0.040, 0.016, 0.013, 0.002]),
        },
        "Centralized": {
            "c_index": metric_entry([0.873, 0.590, 0.666, 0.581, 0.948], [0.016, 0.131, 0.093, 0.181, 0.030]),
            "auc": metric_entry([0.832, 0.600, 0.583, 0.521, 0.951], [0.031, 0.124, 0.129, 0.211, 0.034]),
            "ibs": metric_entry([0.146, 0.965, 0.160, 0.143, 0.038], [0.008, 0.184, 0.016, 0.017, 0.005]),
        },
    },
    4: {
        "Local": {
            "c_index": metric_entry([0.784, 0.350, 0.577, 0.552], [0.045, 0.156, 0.131, 0.221]),
            "auc": metric_entry([0.718, 0.305, 0.640, 0.594], [0.054, 0.158, 0.134, 0.248]),
            "ibs": metric_entry([0.169, 0.314, 0.198, 0.106], [0.018, 0.052, 0.018, 0.019]),
        },
        "Federated": {
            "c_index": metric_entry([0.703, 0.481, 0.666, 0.527], [0.125, 0.241, 0.123, 0.250]),
            "auc": metric_entry([0.659, 0.471, 0.635, 0.560], [0.123, 0.267, 0.142, 0.280]),
            "ibs": metric_entry([0.171, 0.266, 0.192, 0.102], [0.022, 0.042, 0.016, 0.013]),
        },
        "Centralized": {
            "c_index": metric_entry([0.855, 0.631, 0.733, 0.589], [0.028, 0.105, 0.055, 0.183]),
            "auc": metric_entry([0.803, 0.616, 0.661, 0.554], [0.040, 0.105, 0.086, 0.208]),
            "ibs": metric_entry([0.135, 0.769, 0.138, 0.112], [0.011, 0.125, 0.013, 0.014]),
        },
    },
    3: {
        "Local": {
            "c_index": metric_entry([0.784, 0.350, 0.577], [0.045, 0.156, 0.131]),
            "auc": metric_entry([0.718, 0.305, 0.640], [0.054, 0.158, 0.134]),
            "ibs": metric_entry([0.169, 0.314, 0.198], [0.018, 0.052, 0.018]),
        },
        "Federated": {
            "c_index": metric_entry([0.735, 0.445, 0.647], [0.107, 0.218, 0.126]),
            "auc": metric_entry([0.692, 0.408, 0.620], [0.101, 0.234, 0.132]),
            "ibs": metric_entry([0.168, 0.278, 0.192], [0.020, 0.041, 0.015]),
        },
        "Centralized": {
            "c_index": metric_entry([0.861, 0.536, 0.700], [0.020, 0.103, 0.086]),
            "auc": metric_entry([0.807, 0.549, 0.619], [0.030, 0.073, 0.139]),
            "ibs": metric_entry([0.141, 0.810, 0.152], [0.008, 0.153, 0.020]),
        },
    },
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
        "Client-level performance variability for CoxPH",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.95])
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_client_variability()
