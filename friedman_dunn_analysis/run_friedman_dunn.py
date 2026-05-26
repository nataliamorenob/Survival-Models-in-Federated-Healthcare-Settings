#!/usr/bin/env python3
"""Run Friedman + Dunn pairwise analyses over saved experiment CSV files.

The script scans the folders under ``friedman_dunn_analysis`` and, for each
algorithm, evaluates the requested comparisons for the 3, 4 and 5 client
configurations using:

- ``Local``: mean over the local clients involved in the configuration
- ``Centralized``: mean over client-level rows, excluding ``*_global``
- ``FedAvg``: mean over client-level rows from the final federated round

For each metric (AUC, C-Index and IBS), it:

1. builds paired vectors across the 10 runs,
2. runs a Friedman test across ``Local``, ``FedAvg`` and ``Centralized``,
3. if the Friedman test is significant, runs Dunn pairwise tests with
   Holm correction for:
   - Local vs FedAvg
   - Local vs Centralized
   - FedAvg vs Centralized

Outputs:

- ``friedman_dunn_summary.csv``: one row per analysis
- ``friedman_dunn_run_values.csv``: the per-run aggregated values used
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, rankdata

CONFIG_RE = re.compile(r"^(?P<count>\d+)cl$")
CLIENT_DIR_RE = re.compile(r"^c\d+$")
RUN_FILE_RE = re.compile(r"^run_(\d+)\.csv$")

METRICS = ("auc", "c_index", "ibs")
METHODS = ("Local", "FedAvg", "Centralized")
PAIRWISE_COMPARISONS = (
    ("Local", "FedAvg"),
    ("Local", "Centralized"),
    ("FedAvg", "Centralized"),
)
CENTRALIZED_NOTE = (
    "Not statistically significant differences with centralized according to "
    "Dunn's test (Holm-adjusted p > 0.05)"
)
LOGGER = logging.getLogger("friedman_dunn")


@dataclass(frozen=True)
class AnalysisUnit:
    experiment: str
    variant: str
    client_config: str
    num_clients: int
    local_base: Path
    centralized_base: Path
    fedavg_base: Path


class CsvCache:
    def __init__(self) -> None:
        self._cache: dict[Path, pd.DataFrame] = {}

    def read(self, csv_path: Path) -> pd.DataFrame:
        if csv_path not in self._cache:
            LOGGER.debug("Reading CSV from disk: %s", csv_path)
            self._cache[csv_path] = pd.read_csv(csv_path)
        else:
            LOGGER.debug("Using cached CSV: %s", csv_path)
        return self._cache[csv_path].copy()


def numeric_config_sort_key(config_name: str) -> tuple[int, str]:
    match = CONFIG_RE.match(config_name)
    if match is None:
        return (math.inf, config_name)
    return (int(match.group("count")), config_name)


def experiment_directories(root: Path) -> list[Path]:
    if root.is_dir() and root.name.endswith("_allRuns"):
        LOGGER.info("Using single experiment directory as root: %s", root)
        return [root]

    experiment_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.endswith("_allRuns")]
    )
    LOGGER.info("Discovered %d experiment directories under %s", len(experiment_dirs), root)
    for experiment_dir in experiment_dirs:
        LOGGER.debug("Experiment directory: %s", experiment_dir)
    return experiment_dirs


def discover_analysis_units(root: Path) -> list[AnalysisUnit]:
    units: list[AnalysisUnit] = []

    for experiment_dir in experiment_directories(root):
        local_dir = experiment_dir / "Local"
        centralized_dir = experiment_dir / "Centralized"
        federated_dir = experiment_dir / "Federated"
        fedavg_dir = federated_dir / "FedAvg" if (federated_dir / "FedAvg").is_dir() else federated_dir

        if not (local_dir.is_dir() and centralized_dir.is_dir() and fedavg_dir.is_dir()):
            LOGGER.warning(
                "Skipping experiment %s because required folders are missing. local=%s centralized=%s fedavg=%s",
                experiment_dir.name,
                local_dir.is_dir(),
                centralized_dir.is_dir(),
                fedavg_dir.is_dir(),
            )
            continue

        for variant_name, local_base in discover_local_variants(local_dir):
            config_names = discover_config_names(centralized_dir, fedavg_dir, variant_name)
            for config_name in sorted(config_names, key=numeric_config_sort_key):
                match = CONFIG_RE.match(config_name)
                if match is None:
                    continue

                if variant_name == "default":
                    centralized_base = resolve_run_directory(centralized_dir / config_name)
                    fedavg_base = resolve_run_directory(fedavg_dir / config_name)
                else:
                    centralized_base = resolve_run_directory(centralized_dir / config_name / variant_name)
                    fedavg_base = resolve_run_directory(fedavg_dir / config_name / variant_name)

                units.append(
                    AnalysisUnit(
                        experiment=experiment_dir.name,
                        variant=variant_name,
                        client_config=config_name,
                        num_clients=int(match.group("count")),
                        local_base=local_base,
                        centralized_base=centralized_base,
                        fedavg_base=fedavg_base,
                    )
                )
                LOGGER.debug(
                    "Prepared analysis unit: experiment=%s variant=%s config=%s local=%s centralized=%s fedavg=%s",
                    experiment_dir.name,
                    variant_name,
                    config_name,
                    local_base,
                    centralized_base,
                    fedavg_base,
                )

    LOGGER.info("Prepared %d analysis units", len(units))
    return units


def discover_local_variants(local_dir: Path) -> list[tuple[str, Path]]:
    child_dirs = sorted([path for path in local_dir.iterdir() if path.is_dir()])
    if any(CLIENT_DIR_RE.match(path.name) for path in child_dirs):
        LOGGER.debug("Local directory %s uses default client layout", local_dir)
        return [("default", local_dir)]

    variants: list[tuple[str, Path]] = []
    for path in child_dirs:
        if path.name.startswith("."):
            continue
        if any(CLIENT_DIR_RE.match(child.name) for child in path.iterdir() if child.is_dir()):
            variants.append((path.name, path))
            LOGGER.debug("Discovered local variant %s at %s", path.name, path)
    return variants


def discover_config_names(
    centralized_dir: Path,
    fedavg_dir: Path,
    variant_name: str,
) -> set[str]:
    config_names = {
        path.name
        for path in centralized_dir.iterdir()
        if path.is_dir() and CONFIG_RE.match(path.name)
    }

    fedavg_config_names = {
        path.name
        for path in fedavg_dir.iterdir()
        if path.is_dir() and CONFIG_RE.match(path.name)
    }

    shared_configs = config_names & fedavg_config_names
    LOGGER.debug(
        "Config discovery for variant=%s. centralized=%s fedavg=%s shared=%s",
        variant_name,
        sorted(config_names),
        sorted(fedavg_config_names),
        sorted(shared_configs),
    )
    if variant_name == "default":
        result = {
            config
            for config in shared_configs
            if (centralized_dir / config).is_dir() and (fedavg_dir / config).is_dir()
        }
        LOGGER.debug("Usable default configs: %s", sorted(result))
        return result

    result = {
        config
        for config in shared_configs
        if (centralized_dir / config / variant_name).is_dir()
        and (fedavg_dir / config / variant_name).is_dir()
    }
    LOGGER.debug("Usable variant configs for %s: %s", variant_name, sorted(result))
    return result


def has_run_files(directory: Path) -> bool:
    return any(
        path.is_file() and RUN_FILE_RE.match(path.name)
        for path in directory.iterdir()
    )


def resolve_run_directory(directory: Path) -> Path:
    if has_run_files(directory):
        LOGGER.debug("Using run directory directly: %s", directory)
        return directory

    child_dirs = sorted(
        [path for path in directory.iterdir() if path.is_dir() and not path.name.startswith(".")]
    )
    matching_children = [path for path in child_dirs if has_run_files(path)]

    if len(matching_children) == 1:
        LOGGER.info(
            "Resolved nested run directory: %s -> %s",
            directory,
            matching_children[0],
        )
        return matching_children[0]

    LOGGER.debug(
        "Run directory %s has %d nested children with run files; keeping original path",
        directory,
        len(matching_children),
    )
    return directory


def available_run_ids(*directories: Path) -> list[int]:
    shared_run_ids: set[int] | None = None

    for directory in directories:
        run_ids = {
            int(match.group(1))
            for path in directory.iterdir()
            if path.is_file() and (match := RUN_FILE_RE.match(path.name))
        }
        LOGGER.debug("Run IDs in %s: %s", directory, sorted(run_ids))
        if shared_run_ids is None:
            shared_run_ids = run_ids
        else:
            shared_run_ids &= run_ids

    result = sorted(shared_run_ids or [])
    LOGGER.debug("Shared run IDs across directories: %s", result)
    return result


def local_client_dirs(local_base: Path, num_clients: int) -> list[Path]:
    client_dirs = [local_base / f"c{client_idx}" for client_idx in range(num_clients)]
    missing = [path for path in client_dirs if not path.is_dir()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Local client folders: {missing_str}")
    LOGGER.debug("Local client directories for %s (%d clients): %s", local_base, num_clients, client_dirs)
    return client_dirs


def mean_without_nan(values: Iterable[float], context: str) -> float:
    raw_values = list(values)
    numeric = pd.to_numeric(pd.Series(raw_values), errors="coerce").dropna()
    if numeric.empty:
        raise ValueError(f"No valid numeric values found for {context}")
    mean_value = float(numeric.mean())
    LOGGER.debug(
        "Mean calculation for %s | raw_values=%s | numeric_values=%s | mean=%s",
        context,
        raw_values,
        numeric.tolist(),
        mean_value,
    )
    return mean_value


def aggregate_local_metric(
    cache: CsvCache,
    local_base: Path,
    num_clients: int,
    run_id: int,
    metric: str,
) -> float:
    values = []
    per_client_values: dict[str, list[float]] = {}
    for client_dir in local_client_dirs(local_base, num_clients):
        csv_path = client_dir / f"run_{run_id}.csv"
        df = cache.read(csv_path)
        client_values = df[metric].tolist()
        per_client_values[client_dir.name] = client_values
        values.extend(client_values)
    mean_value = mean_without_nan(values, f"Local {metric} in {local_base} run {run_id}")
    LOGGER.debug(
        "Aggregated Local metric | base=%s | run=%s | metric=%s | per_client_values=%s | aggregated_mean=%s",
        local_base,
        run_id,
        metric,
        per_client_values,
        mean_value,
    )
    return mean_value


def aggregate_centralized_metric(
    cache: CsvCache,
    centralized_base: Path,
    run_id: int,
    metric: str,
) -> float:
    csv_path = centralized_base / f"run_{run_id}.csv"
    df = cache.read(csv_path)
    client_rows = df[
        ~df["client_id"].astype(str).str.contains("global", case=False, na=False)
    ]
    mean_value = mean_without_nan(
        client_rows[metric].tolist(),
        f"Centralized {metric} in {centralized_base} run {run_id}",
    )
    LOGGER.debug(
        "Aggregated Centralized metric | base=%s | run=%s | metric=%s | used_rows=%s | aggregated_mean=%s",
        centralized_base,
        run_id,
        metric,
        client_rows[["client_id", metric]].to_dict(orient="records"),
        mean_value,
    )
    return mean_value


def aggregate_fedavg_metric(
    cache: CsvCache,
    fedavg_base: Path,
    run_id: int,
    metric: str,
) -> float:
    csv_path = fedavg_base / f"run_{run_id}.csv"
    df = cache.read(csv_path)
    if "round" not in df.columns:
        raise KeyError(f"Expected a 'round' column in {csv_path}")

    numeric_rounds = pd.to_numeric(df["round"], errors="coerce")
    final_round = numeric_rounds.max()
    final_round_rows = df[numeric_rounds == final_round]
    mean_value = mean_without_nan(
        final_round_rows[metric].tolist(),
        f"FedAvg {metric} in {fedavg_base} run {run_id} final round {final_round}",
    )
    LOGGER.debug(
        "Aggregated FedAvg metric | base=%s | run=%s | metric=%s | rounds_present=%s | final_round=%s | used_rows=%s | aggregated_mean=%s",
        fedavg_base,
        run_id,
        metric,
        sorted(numeric_rounds.dropna().astype(int).unique().tolist()),
        final_round,
        final_round_rows[["round", "client_id", metric]].to_dict(orient="records"),
        mean_value,
    )
    return mean_value


def holm_adjust(p_values: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    sorted_items = sorted(p_values.items(), key=lambda item: item[1])
    m = len(sorted_items)

    adjusted: dict[tuple[str, str], float] = {}
    running_max = 0.0
    for rank, (pair, p_value) in enumerate(sorted_items, start=1):
        adjusted_value = min(1.0, p_value * (m - rank + 1))
        running_max = max(running_max, adjusted_value)
        adjusted[pair] = running_max

    LOGGER.debug(
        "Holm adjustment | raw_p_values=%s | sorted=%s | adjusted=%s",
        p_values,
        sorted_items,
        adjusted,
    )
    return adjusted


def dunn_pairwise(group_values: dict[str, list[float]]) -> dict[tuple[str, str], dict[str, float]]:
    values: list[float] = []
    labels: list[str] = []
    group_sizes: dict[str, int] = {}

    for method_name in METHODS:
        method_values = group_values[method_name]
        values.extend(method_values)
        labels.extend([method_name] * len(method_values))
        group_sizes[method_name] = len(method_values)

    values_array = np.asarray(values, dtype=float)
    ranks = rankdata(values_array, method="average")

    unique_values, tie_counts = np.unique(values_array, return_counts=True)
    _ = unique_values  # explicit to show the values are intentionally unused
    tie_sum = float(np.sum(tie_counts**3 - tie_counts))
    n_total = len(values_array)
    tie_term = tie_sum / (12.0 * (n_total - 1)) if n_total > 1 else 0.0
    base_variance = (n_total * (n_total + 1) / 12.0) - tie_term

    mean_ranks: dict[str, float] = {}
    labels_array = np.asarray(labels)
    for method_name in METHODS:
        mean_ranks[method_name] = float(ranks[labels_array == method_name].mean())

    raw_p_values: dict[tuple[str, str], float] = {}
    z_scores: dict[tuple[str, str], float] = {}

    for method_a, method_b in PAIRWISE_COMPARISONS:
        denominator = math.sqrt(
            base_variance
            * ((1.0 / group_sizes[method_a]) + (1.0 / group_sizes[method_b]))
        )
        z_value = (mean_ranks[method_a] - mean_ranks[method_b]) / denominator
        p_value = float(2.0 * norm.sf(abs(z_value)))
        z_scores[(method_a, method_b)] = z_value
        raw_p_values[(method_a, method_b)] = p_value

    adjusted_p_values = holm_adjust(raw_p_values)

    results = {
        pair: {
            "z_score": z_scores[pair],
            "p_raw": raw_p_values[pair],
            "p_holm": adjusted_p_values[pair],
        }
        for pair in PAIRWISE_COMPARISONS
    }
    LOGGER.debug(
        "Dunn test details | group_values=%s | values_array=%s | labels=%s | ranks=%s | tie_counts=%s | tie_sum=%s | n_total=%s | tie_term=%s | base_variance=%s | mean_ranks=%s | z_scores=%s | raw_p_values=%s | adjusted_results=%s",
        group_values,
        values_array.tolist(),
        labels,
        ranks.tolist(),
        tie_counts.tolist(),
        tie_sum,
        n_total,
        tie_term,
        base_variance,
        mean_ranks,
        z_scores,
        raw_p_values,
        results,
    )
    return results


def run_analysis(root: Path, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = CsvCache()
    summary_rows: list[dict[str, object]] = []
    run_value_rows: list[dict[str, object]] = []

    for unit in discover_analysis_units(root):
        LOGGER.info(
            "Starting analysis unit | experiment=%s | variant=%s | client_config=%s | num_clients=%d",
            unit.experiment,
            unit.variant,
            unit.client_config,
            unit.num_clients,
        )
        try:
            local_dirs = local_client_dirs(unit.local_base, unit.num_clients)
        except FileNotFoundError as exc:
            LOGGER.warning(
                "Skipping %s | %s | %s: %s",
                unit.experiment,
                unit.variant,
                unit.client_config,
                exc,
            )
            continue

        run_ids = available_run_ids(
            *local_dirs,
            unit.centralized_base,
            unit.fedavg_base,
        )

        if not run_ids:
            LOGGER.warning(
                "Skipping %s | %s | %s: no shared run_*.csv files found across Local, Centralized and FedAvg.",
                unit.experiment,
                unit.variant,
                unit.client_config,
            )
            continue
        LOGGER.info(
            "Shared run IDs for unit %s | %s | %s: %s",
            unit.experiment,
            unit.variant,
            unit.client_config,
            run_ids,
        )

        for metric in METRICS:
            grouped_values = {method: [] for method in METHODS}
            LOGGER.info(
                "Computing metric %s for %s | %s | %s",
                metric,
                unit.experiment,
                unit.variant,
                unit.client_config,
            )

            for run_id in run_ids:
                local_value = aggregate_local_metric(
                    cache=cache,
                    local_base=unit.local_base,
                    num_clients=unit.num_clients,
                    run_id=run_id,
                    metric=metric,
                )
                fedavg_value = aggregate_fedavg_metric(
                    cache=cache,
                    fedavg_base=unit.fedavg_base,
                    run_id=run_id,
                    metric=metric,
                )
                centralized_value = aggregate_centralized_metric(
                    cache=cache,
                    centralized_base=unit.centralized_base,
                    run_id=run_id,
                    metric=metric,
                )

                grouped_values["Local"].append(local_value)
                grouped_values["FedAvg"].append(fedavg_value)
                grouped_values["Centralized"].append(centralized_value)
                LOGGER.debug(
                    "Per-run aggregated values | experiment=%s | variant=%s | config=%s | metric=%s | run=%s | Local=%s | FedAvg=%s | Centralized=%s",
                    unit.experiment,
                    unit.variant,
                    unit.client_config,
                    metric,
                    run_id,
                    local_value,
                    fedavg_value,
                    centralized_value,
                )

                run_value_rows.append(
                    {
                        "experiment": unit.experiment,
                        "variant": unit.variant,
                        "client_config": unit.client_config,
                        "num_clients": unit.num_clients,
                        "metric": metric,
                        "run_id": run_id,
                        "local": local_value,
                        "fedavg": fedavg_value,
                        "centralized": centralized_value,
                    }
                )

            friedman_statistic, friedman_p_value = friedmanchisquare(
                grouped_values["Local"],
                grouped_values["FedAvg"],
                grouped_values["Centralized"],
            )
            LOGGER.info(
                "Friedman test | experiment=%s | variant=%s | config=%s | metric=%s | Local=%s | FedAvg=%s | Centralized=%s | statistic=%s | p_value=%s",
                unit.experiment,
                unit.variant,
                unit.client_config,
                metric,
                grouped_values["Local"],
                grouped_values["FedAvg"],
                grouped_values["Centralized"],
                float(friedman_statistic),
                float(friedman_p_value),
            )

            pairwise_results: dict[tuple[str, str], dict[str, float]] = {}
            if friedman_p_value < alpha:
                LOGGER.info(
                    "Friedman significant for %s | %s | %s | %s. Running Dunn pairwise comparisons.",
                    unit.experiment,
                    unit.variant,
                    unit.client_config,
                    metric,
                )
                pairwise_results = dunn_pairwise(grouped_values)
            else:
                LOGGER.info(
                    "Friedman not significant for %s | %s | %s | %s. Dunn pairwise comparisons skipped.",
                    unit.experiment,
                    unit.variant,
                    unit.client_config,
                    metric,
                )

            local_vs_centralized = pairwise_results.get(("Local", "Centralized"))
            fedavg_vs_centralized = pairwise_results.get(("FedAvg", "Centralized"))

            local_not_diff_from_centralized = bool(
                local_vs_centralized and local_vs_centralized["p_holm"] > alpha
            )
            fedavg_not_diff_from_centralized = bool(
                fedavg_vs_centralized and fedavg_vs_centralized["p_holm"] > alpha
            )

            summary_rows.append(
                {
                    "experiment": unit.experiment,
                    "variant": unit.variant,
                    "client_config": unit.client_config,
                    "num_clients": unit.num_clients,
                    "metric": metric,
                    "n_runs_used": len(run_ids),
                    "friedman_statistic": float(friedman_statistic),
                    "friedman_p_value": float(friedman_p_value),
                    "friedman_significant": bool(friedman_p_value < alpha),
                    "local_mean": float(np.mean(grouped_values["Local"])),
                    "fedavg_mean": float(np.mean(grouped_values["FedAvg"])),
                    "centralized_mean": float(np.mean(grouped_values["Centralized"])),
                    "local_vs_fedavg_z": pairwise_results.get(("Local", "FedAvg"), {}).get("z_score"),
                    "local_vs_fedavg_p_raw": pairwise_results.get(("Local", "FedAvg"), {}).get("p_raw"),
                    "local_vs_fedavg_p_holm": pairwise_results.get(("Local", "FedAvg"), {}).get("p_holm"),
                    "local_vs_centralized_z": local_vs_centralized.get("z_score") if local_vs_centralized else None,
                    "local_vs_centralized_p_raw": local_vs_centralized.get("p_raw") if local_vs_centralized else None,
                    "local_vs_centralized_p_holm": local_vs_centralized.get("p_holm") if local_vs_centralized else None,
                    "fedavg_vs_centralized_z": fedavg_vs_centralized.get("z_score") if fedavg_vs_centralized else None,
                    "fedavg_vs_centralized_p_raw": fedavg_vs_centralized.get("p_raw") if fedavg_vs_centralized else None,
                    "fedavg_vs_centralized_p_holm": fedavg_vs_centralized.get("p_holm") if fedavg_vs_centralized else None,
                    "local_not_significantly_different_from_centralized": local_not_diff_from_centralized,
                    "fedavg_not_significantly_different_from_centralized": fedavg_not_diff_from_centralized,
                    "local_asterisk": "*" if local_not_diff_from_centralized else "",
                    "fedavg_asterisk": "*" if fedavg_not_diff_from_centralized else "",
                    "local_note": CENTRALIZED_NOTE if local_not_diff_from_centralized else "",
                    "fedavg_note": CENTRALIZED_NOTE if fedavg_not_diff_from_centralized else "",
                }
            )
            LOGGER.info(
                "Summary row saved | experiment=%s | variant=%s | config=%s | metric=%s | friedman_significant=%s | pairwise_results=%s",
                unit.experiment,
                unit.variant,
                unit.client_config,
                metric,
                bool(friedman_p_value < alpha),
                pairwise_results,
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["experiment", "variant", "num_clients", "metric"]
    )
    run_values_df = pd.DataFrame(run_value_rows).sort_values(
        by=["experiment", "variant", "num_clients", "metric", "run_id"]
    )
    return summary_df, run_values_df


def compact_result_label(value: bool) -> str:
    return "Yes" if bool(value) else "No"


def compact_ns_label(flag_value: object, p_value: object) -> str:
    if pd.isna(p_value):
        return "N/A"
    return "Yes" if bool(flag_value) else "No"


def build_compact_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    compact_df = pd.DataFrame(
        {
            "experiment": summary_df["experiment"],
            "variant": summary_df["variant"],
            "client_config": summary_df["client_config"],
            "metric": summary_df["metric"],
            "friedman_p_value": summary_df["friedman_p_value"],
            "friedman_significant": summary_df["friedman_significant"].map(compact_result_label),
            "local_vs_centralized_p_holm": summary_df["local_vs_centralized_p_holm"],
            "local_ns_vs_centralized": [
                compact_ns_label(flag, p_value)
                for flag, p_value in zip(
                    summary_df["local_not_significantly_different_from_centralized"],
                    summary_df["local_vs_centralized_p_holm"],
                )
            ],
            "fedavg_vs_centralized_p_holm": summary_df["fedavg_vs_centralized_p_holm"],
            "fedavg_ns_vs_centralized": [
                compact_ns_label(flag, p_value)
                for flag, p_value in zip(
                    summary_df["fedavg_not_significantly_different_from_centralized"],
                    summary_df["fedavg_vs_centralized_p_holm"],
                )
            ],
            "local_mark": summary_df["local_asterisk"].fillna(""),
            "fedavg_mark": summary_df["fedavg_asterisk"].fillna(""),
        }
    )
    return compact_df


def write_markdown_table(df: pd.DataFrame, output_path: Path) -> None:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        formatted_cells = []
        for value in row.tolist():
            if pd.isna(value):
                formatted_cells.append("")
            else:
                formatted_cells.append(str(value))
        lines.append("| " + " | ".join(formatted_cells) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def setup_logging(log_file: Path, console_level: str) -> None:
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    stream_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Friedman and Dunn analyses over saved result CSVs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Analysis root folder. Defaults to the folder containing this script.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level used for Friedman and Dunn decisions.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Detailed log file path. Defaults to <root>/friedman_dunn_detailed.log.",
    )
    parser.add_argument(
        "--console-log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console logging level. Detailed logs are always written to the log file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = args.root.resolve()
    alpha = args.alpha
    log_file = args.log_file.resolve() if args.log_file else (analysis_root / "friedman_dunn_detailed.log")

    setup_logging(log_file=log_file, console_level=args.console_log_level)
    LOGGER.info("Starting Friedman/Dunn analysis")
    LOGGER.info("Analysis root: %s", analysis_root)
    LOGGER.info("Alpha: %s", alpha)
    LOGGER.info("Detailed log file: %s", log_file)

    summary_df, run_values_df = run_analysis(analysis_root, alpha=alpha)
    compact_summary_df = build_compact_summary(summary_df)

    summary_path = analysis_root / "friedman_dunn_summary.csv"
    run_values_path = analysis_root / "friedman_dunn_run_values.csv"
    compact_summary_path = analysis_root / "friedman_dunn_summary_compact.csv"
    compact_summary_md_path = analysis_root / "friedman_dunn_summary_compact.md"

    summary_df.to_csv(summary_path, index=False)
    run_values_df.to_csv(run_values_path, index=False)
    compact_summary_df.to_csv(compact_summary_path, index=False)
    write_markdown_table(compact_summary_df, compact_summary_md_path)

    LOGGER.info("Saved summary results to: %s", summary_path)
    LOGGER.info("Saved per-run aggregated values to: %s", run_values_path)
    LOGGER.info("Saved compact summary results to: %s", compact_summary_path)
    LOGGER.info("Saved compact summary markdown to: %s", compact_summary_md_path)
    LOGGER.info("Finished Friedman/Dunn analysis")


if __name__ == "__main__":
    main()
