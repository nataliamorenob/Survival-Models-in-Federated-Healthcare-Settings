"""
Create Figure 1 for the paper.

Figure layout:
- 3 subplots: C-index, AUC, IBS
- x-axis: number of clients
- y-axis: metric value
- lines: Local, Federated, Centralized
- colors/markers: CoxPH, DeepSurv, RSF
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
from matplotlib.lines import Line2D


plt.style.use("seaborn-v0_8-whitegrid")

OUTPUT_PATH = Path(__file__).with_name("figure_1_multimodel_client_counts.png")

CLIENT_COUNTS = [3, 4, 5]
METRIC_SPECS = [
    ("c_index", "C-index"),
    ("auc", "AUC"),
    ("ibs", "IBS"),
]
PARADIGMS = ["Local", "Federated", "Centralized"]

MODEL_STYLES = {
    "CoxPH": {"color": "#D55E00", "marker": "o"},
    "DeepSurv": {"color": "#0072B2", "marker": "s"},
    "RSF": {"color": "#009E73", "marker": "^"},
}

PARADIGM_LINESTYLES = {
    "Local": "-",
    "Federated": "--",
    "Centralized": ":",
}

TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 11
LEGEND_FONT_SIZE = 11

RESULTS = {
    "CoxPH": {
        "Local": {
            5: {"c_index": 0.598, "auc": 0.606, "ibs": 0.167},
            4: {"c_index": 0.566, "auc": 0.564, "ibs": 0.197},
            3: {"c_index": 0.570, "auc": 0.554, "ibs": 0.227},
        },
        "Federated": {
            5: {"c_index": 0.611, "auc": 0.600, "ibs": 0.155},
            4: {"c_index": 0.594, "auc": 0.581, "ibs": 0.183},
            3: {"c_index": 0.609, "auc": 0.573, "ibs": 0.213},
        },
        "Centralized": {
            5: {"c_index": 0.732, "auc": 0.697, "ibs": 0.290},
            4: {"c_index": 0.702, "auc": 0.659, "ibs": 0.288},
            3: {"c_index": 0.699, "auc": 0.658, "ibs": 0.368},
        },
    },
    "DeepSurv": {
        "Local": {
            5: {"c_index": 0.618, "auc": 0.620, "ibs": 0.159},
            4: {"c_index": 0.578, "auc": 0.573, "ibs": 0.187},
            3: {"c_index": 0.601, "auc": 0.595, "ibs": 0.215},
        },
        "Federated": {
            5: {"c_index": 0.727, "auc": 0.702, "ibs": 0.148},
            4: {"c_index": 0.671, "auc": 0.636, "ibs": 0.174},
            3: {"c_index": 0.724, "auc": 0.667, "ibs": 0.198},
        },
        "Centralized": {
            5: {"c_index": 0.707, "auc": 0.684, "ibs": 0.269},
            4: {"c_index": 0.677, "auc": 0.651, "ibs": 0.288},
            3: {"c_index": 0.650, "auc": 0.616, "ibs": 0.347},
        },
    },
    "RSF": {
        "Local": {
            5: {"c_index": 0.687, "auc": 0.665, "ibs": 0.152},
            4: {"c_index": 0.615, "auc": 0.594, "ibs": 0.180},
            3: {"c_index": 0.617, "auc": 0.580, "ibs": 0.207},
        },
        "Federated": {
            5: {"c_index": 0.724, "auc": 0.678, "ibs": 0.138},
            4: {"c_index": 0.678, "auc": 0.640, "ibs": 0.159},
            3: {"c_index": 0.690, "auc": 0.600, "ibs": 0.181},
        },
        "Centralized": {
            5: {"c_index": 0.746, "auc": 0.720, "ibs": 0.245},
            4: {"c_index": 0.722, "auc": 0.688, "ibs": 0.271},
            3: {"c_index": 0.724, "auc": 0.648, "ibs": 0.318},
        },
    },
}


def compute_ylim(metric_name: str) -> tuple[float, float]:
    values = [
        RESULTS[model][paradigm][client_count][metric_name]
        for model in RESULTS
        for paradigm in PARADIGMS
        for client_count in CLIENT_COUNTS
    ]
    min_value = min(values)
    max_value = max(values)
    padding = 0.04 if metric_name != "ibs" else 0.03
    return max(0.0, min_value - padding), min(1.0, max_value + padding)


def plot_figure_1() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    x = np.array(CLIENT_COUNTS, dtype=float)

    for ax, (metric_key, metric_label) in zip(axes, METRIC_SPECS):
        for model_name, style in MODEL_STYLES.items():
            for paradigm in PARADIGMS:
                y = [
                    RESULTS[model_name][paradigm][client_count][metric_key]
                    for client_count in CLIENT_COUNTS
                ]
                ax.plot(
                    x,
                    y,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=PARADIGM_LINESTYLES[paradigm],
                    linewidth=2.0,
                    markersize=7,
                    alpha=0.95,
                )

        ax.set_title(metric_label, fontsize=TITLE_FONT_SIZE, fontweight="bold")
        ax.set_xlabel("Number of clients", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Metric value", fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(CLIENT_COUNTS)
        ax.set_xlim(2.8, 5.2)
        ax.set_ylim(*compute_ylim(metric_key))
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2.0,
            markersize=7,
            label=model_name,
        )
        for model_name, style in MODEL_STYLES.items()
    ]
    paradigm_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=PARADIGM_LINESTYLES[paradigm],
            linewidth=2.0,
            label=paradigm,
        )
        for paradigm in PARADIGMS
    ]

    model_legend = fig.legend(
        handles=model_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.35, 1.02),
        fontsize=LEGEND_FONT_SIZE,
        title="Model",
        title_fontsize=LEGEND_FONT_SIZE,
    )
    fig.add_artist(model_legend)
    fig.legend(
        handles=paradigm_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.79, 1.02),
        fontsize=LEGEND_FONT_SIZE,
        title="Training",
        title_fontsize=LEGEND_FONT_SIZE,
    )

    fig.suptitle(
        "Performance across paradigms and number of clients",
        fontsize=17,
        fontweight="bold",
        y=1.10,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.90])
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_figure_1()
