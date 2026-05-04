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

OUTPUT_PATH = Path(__file__).with_name("figure_2_CoxPH.png")
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

TITLE_FONT_SIZE = 17
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 11.5
SECTION_LABEL_FONT_SIZE = 15


def metric_entry(means: list[float], stds: list[float]) -> dict[str, list[float]]:
    return {"means": means, "stds": stds}


RAW_RESULTS = {
    5: {
        "Local": {
            "c_index": metric_entry([0.784, 0.350, 0.577, 0.552, 0.725], [0.045, 0.156, 0.131, 0.221, 0.254]),
            "auc": metric_entry([0.718, 0.305, 0.640, 0.594, 0.771], [0.054, 0.158, 0.134, 0.248, 0.275]),
            "ibs": metric_entry([0.169, 0.314, 0.198, 0.106, 0.047], [0.018, 0.052, 0.018, 0.019, 0.004]),
        },
        "Federated": {
            "FedAvg": {
                "c_index": metric_entry([0.712, 0.481, 0.646, 0.514, 0.702], [0.139, 0.218, 0.129, 0.231, 0.234]),
                "auc": metric_entry([0.669, 0.476, 0.598, 0.554, 0.705], [0.137, 0.240, 0.154, 0.268, 0.263]),
                "ibs": metric_entry([0.169, 0.262, 0.194, 0.103, 0.049], [0.023, 0.040, 0.016, 0.013, 0.002]),
            }
        },
    },
    3: {
        "Local": {
            "c_index": metric_entry([0.784, 0.350, 0.577], [0.045, 0.156, 0.131]),
            "auc": metric_entry([0.718, 0.305, 0.640], [0.054, 0.158, 0.134]),
            "ibs": metric_entry([0.169, 0.314, 0.198], [0.018, 0.052, 0.018]),
        },
        "Federated": {
            "FedAvg": {
                "c_index": metric_entry([0.735, 0.445, 0.647], [0.107, 0.218, 0.126]),
                "auc": metric_entry([0.692, 0.408, 0.620], [0.101, 0.234, 0.132]),
                "ibs": metric_entry([0.168, 0.278, 0.192], [0.020, 0.041, 0.015]),
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
            fontsize=ANNOTATION_FONT_SIZE,
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
            ax.set_xticklabels(clients, fontsize=TICK_FONT_SIZE)
            ax.set_xlabel("Clients", fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel("Delta (Federated - Local)", fontsize=LABEL_FONT_SIZE)
            ax.set_title(metric_label, fontsize=TITLE_FONT_SIZE, fontweight="bold")
            ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
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
                    fontsize=SECTION_LABEL_FONT_SIZE,
                    fontweight="bold",
                )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_client_level_effect()
