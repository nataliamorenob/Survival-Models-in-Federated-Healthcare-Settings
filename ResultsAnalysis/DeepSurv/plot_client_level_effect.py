"""
Figure 2: client-level effect.

Design:
- X-axis: clients
- Y-axis: delta = Federated - Local
- Type: bar plot
- Strategy: FedAvg only
- Default configuration: 5 clients

Set CLIENT_COUNTS_TO_PLOT = [5, 3] if you also want the 3-client version.
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

OUTPUT_PATH = Path(__file__).with_name("figure_2_DeepSurv.png")
CLIENT_COUNTS_TO_PLOT = [5]

METRIC_SPECS = [
    ("c_index", "C-index"),
    ("auc", "AUC"),
    ("ibs", "IBS"),
]

LOCAL_COLOR = "#D55E00"  # reddish-orange
FEDERATED_COLOR = "#0072B2"  # blue
CENTRALIZED_COLOR = "#009E73"  # bluish-green

POSITIVE_COLOR = FEDERATED_COLOR
NEGATIVE_COLOR = LOCAL_COLOR


def metric_entry(means: list[float], stds: list[float]) -> dict[str, list[float]]:
    return {"means": means, "stds": stds}


RAW_RESULTS = {
    5: {
        "Local": {
            "c_index": metric_entry([0.740, 0.521, 0.543, 0.509, 0.775], [0.130, 0.166, 0.118, 0.119, 0.161]),
            "auc": metric_entry([0.689, 0.515, 0.582, 0.510, 0.807], [0.102, 0.163, 0.120, 0.124, 0.182]),
            "ibs": metric_entry([0.179, 0.270, 0.195, 0.103, 0.046], [0.015, 0.047, 0.011, 0.007, 0.005]),
        },
        "Federated": {
            "FedAvg": {
                "c_index": metric_entry([0.804, 0.640, 0.712, 0.551, 0.929], [0.055, 0.192, 0.090, 0.183, 0.063]),
                "auc": metric_entry([0.763, 0.639, 0.606, 0.569, 0.934], [0.054, 0.205, 0.124, 0.208, 0.112]),
                "ibs": metric_entry([0.162, 0.243, 0.189, 0.101, 0.045], [0.011, 0.030, 0.015, 0.007, 0.004]),
            }
        },
    },
    3: {
        "Local": {
            "c_index": metric_entry([0.740, 0.521, 0.543], [0.130, 0.166, 0.118]),
            "auc": metric_entry([0.689, 0.515, 0.582], [0.102, 0.163, 0.120]),
            "ibs": metric_entry([0.179, 0.270, 0.195], [0.015, 0.047, 0.011]),
        },
        "Federated": {
            "FedAvg": {
                "c_index": metric_entry([0.808, 0.610, 0.754], [0.025, 0.166, 0.063]),
                "auc": metric_entry([0.751, 0.592, 0.657], [0.024, 0.194, 0.117]),
                "ibs": metric_entry([0.165, 0.249, 0.180], [0.013, 0.040, 0.015]),
            }
        },
    },
}


def compute_delta(client_count: int, metric_name: str) -> tuple[np.ndarray, np.ndarray]:
    fed_data = RAW_RESULTS[client_count]["Federated"]["FedAvg"][metric_name]
    local_data = RAW_RESULTS[client_count]["Local"][metric_name]

    fed_means = np.array(fed_data["means"], dtype=float)
    local_means = np.array(local_data["means"], dtype=float)
    fed_stds = np.array(fed_data["stds"], dtype=float)
    local_stds = np.array(local_data["stds"], dtype=float)

    deltas = fed_means - local_means
    delta_stds = np.sqrt(fed_stds**2 + local_stds**2)
    return deltas, delta_stds


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
            fontsize=9.6,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 0.2,
            },
        )


def plot_client_level_effect() -> None:
    n_rows = len(CLIENT_COUNTS_TO_PLOT)
    fig, axes = plt.subplots(n_rows, len(METRIC_SPECS), figsize=(15, 4.5 * n_rows), squeeze=False)

    for row_index, client_count in enumerate(CLIENT_COUNTS_TO_PLOT):
        clients = [f"C{i}" for i in range(client_count)]

        for col_index, (metric_key, metric_label) in enumerate(METRIC_SPECS):
            ax = axes[row_index, col_index]
            deltas, delta_stds = compute_delta(client_count, metric_key)
            x = np.arange(client_count)
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
            ax.set_xticklabels(clients, fontsize=12.6)
            ax.set_xlabel("Clients", fontsize=12.6)
            ax.set_ylabel("Delta (Federated - Local)", fontsize=12.6)
            ax.set_title(metric_label, fontsize=14.6, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.set_axisbelow(True)

            limit = max(np.max(np.abs(deltas + delta_stds)), np.max(np.abs(deltas - delta_stds)))
            ax.set_ylim(-(limit + 0.12), limit + 0.12)
            add_value_labels(ax, bars, deltas)

            if n_rows > 1 and col_index == 0:
                ax.text(
                    -0.38,
                    1.06,
                    f"{client_count} clients",
                    transform=ax.transAxes,
                    fontsize=13.6,
                    fontweight="bold",
                )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_client_level_effect()
