"""
Federated comparison plot for CoxPH, DeepSurv, and RSF.

The script creates a three-panel figure with client-wise grouped bar charts for:
- C-index
- AUC
- IBS

All values are taken from the provided federated experiments.
"""

from __future__ import annotations

import os
from pathlib import Path

MPL_CONFIG_DIR = Path(__file__).with_name(".mplconfig")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


plt.style.use("seaborn-v0_8-whitegrid")

OUTPUT_PATH = Path(__file__).with_name("federated_coxph_vs_deepsurv_vs_rsf.png")

CLIENTS = ["C0", "C1", "C2", "C3", "C4"]
MODELS = ["RSF", "DeepSurv", "CoxPH"]

# Okabe-Ito palette: color-blind-friendly and high contrast.
COLORS = {
    "RSF": "#0072B2",       # blue
    "DeepSurv": "#E69F00",  # orange
    "CoxPH": "#009E73",     # bluish green
}

METRIC_SPECS = [
    ("c_index", "C-index"),
    ("auc", "AUC"),
    ("ibs", "IBS"),
]

RESULTS = {
    "RSF": {
        "C0": {"c_index": (0.821, 0.020), "auc": (0.765, 0.026), "ibs": (0.163, 0.007)},
        "C1": {"c_index": (0.520, 0.130), "auc": (0.443, 0.146), "ibs": (0.191, 0.018)},
        "C2": {"c_index": (0.684, 0.057), "auc": (0.512, 0.092), "ibs": (0.191, 0.010)},
        "C3": {"c_index": (0.658, 0.070), "auc": (0.731, 0.046), "ibs": (0.096, 0.006)},
        "C4": {"c_index": (0.938, 0.051), "auc": (0.937, 0.032), "ibs": (0.047, 0.001)},
    },
    "DeepSurv": {
        "C0": {"c_index": (0.804, 0.055), "auc": (0.763, 0.054), "ibs": (0.162, 0.011)},
        "C1": {"c_index": (0.640, 0.192), "auc": (0.639, 0.205), "ibs": (0.243, 0.030)},
        "C2": {"c_index": (0.712, 0.090), "auc": (0.606, 0.124), "ibs": (0.189, 0.015)},
        "C3": {"c_index": (0.551, 0.183), "auc": (0.569, 0.208), "ibs": (0.101, 0.007)},
        "C4": {"c_index": (0.929, 0.063), "auc": (0.934, 0.112), "ibs": (0.045, 0.004)},
    },
    "CoxPH": {
        "C0": {"c_index": (0.712, 0.139), "auc": (0.669, 0.137), "ibs": (0.169, 0.023)},
        "C1": {"c_index": (0.481, 0.218), "auc": (0.476, 0.240), "ibs": (0.262, 0.040)},
        "C2": {"c_index": (0.646, 0.129), "auc": (0.598, 0.154), "ibs": (0.194, 0.016)},
        "C3": {"c_index": (0.514, 0.231), "auc": (0.554, 0.268), "ibs": (0.103, 0.013)},
        "C4": {"c_index": (0.702, 0.234), "auc": (0.705, 0.263), "ibs": (0.049, 0.002)},
    },
}


def collect_metric_values(metric_key: str) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    means_by_model: dict[str, list[float]] = {}
    stds_by_model: dict[str, list[float]] = {}

    for model in MODELS:
        means_by_model[model] = [RESULTS[model][client][metric_key][0] for client in CLIENTS]
        stds_by_model[model] = [RESULTS[model][client][metric_key][1] for client in CLIENTS]

    return means_by_model, stds_by_model


def add_bar_labels(ax: plt.Axes, bars, values: list[float], stds: list[float]) -> None:
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.03

    for bar, value, std in zip(bars, values, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + std + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )


def plot_federated_model_comparison() -> None:
    x = np.arange(len(CLIENTS))
    width = 0.18
    offsets = {
        "RSF": -0.24,
        "DeepSurv": 0.0,
        "CoxPH": 0.24,
    }

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2))

    for ax, (metric_key, metric_label) in zip(axes, METRIC_SPECS):
        means_by_model, stds_by_model = collect_metric_values(metric_key)
        all_values = []
        all_stds = []

        for model in MODELS:
            means = means_by_model[model]
            stds = stds_by_model[model]
            bars = ax.bar(
                x + offsets[model],
                means,
                width=width,
                yerr=stds,
                capsize=4,
                color=COLORS[model],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.92,
            )
            add_bar_labels(ax, bars, means, stds)
            all_values.extend(means)
            all_stds.extend(stds)

        ax.set_title(metric_label, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CLIENTS, fontsize=11)
        ax.set_xlabel("Client", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        upper_limit = max(value + std for value, std in zip(all_values, all_stds))
        ax.set_ylim(0, min(1.2, upper_limit + 0.22))

    legend_handles = [
        Patch(facecolor=COLORS[model], edgecolor="black", label=model)
        for model in MODELS
    ]

    fig.legend(
        legend_handles,
        MODELS,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.86, wspace=0.28)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

    print(f"Saved figure to: {OUTPUT_PATH}")
    for metric_key, metric_label in METRIC_SPECS:
        print(f"{metric_label}:")
        for model in MODELS:
            values_text = ", ".join(
                f"{client}={RESULTS[model][client][metric_key][0]:.3f}"
                for client in CLIENTS
            )
            print(f"  - {model}: {values_text}")


if __name__ == "__main__":
    plot_federated_model_comparison()
