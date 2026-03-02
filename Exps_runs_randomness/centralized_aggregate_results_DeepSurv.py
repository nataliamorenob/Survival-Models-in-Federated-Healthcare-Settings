import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
RESULTS_DIR = Path("/scratch/project_2015651/Masters_thesis/results_randomness_exps")

N_RUNS = 6

OUTPUT_CLIENT_CSV = RESULTS_DIR / "summary_per_client_mean_std.csv"
OUTPUT_GLOBAL_CSV = RESULTS_DIR / "summary_global_mean_std.csv"

# ---------------------------------------------------------
# LOAD ALL RUN CSVs
# ---------------------------------------------------------
dfs = []

for i in range(1, N_RUNS + 1):
    csv_path = RESULTS_DIR / f"run_{i}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    dfs.append(pd.read_csv(csv_path))

df_all = pd.concat(dfs, ignore_index=True)

print("Loaded total rows:", len(df_all))

# ---------------------------------------------------------
# REMOVE centralized_global
# ---------------------------------------------------------
df_all = df_all[df_all["client_id"] != "centralized_global"]

print("Rows after removing centralized_global:", len(df_all))
print(df_all.head(), "\n")

# ---------------------------------------------------------
# 1) PER-CLIENT: mean ± std over runs
# ---------------------------------------------------------
summary_per_client = (
    df_all
    .groupby("client_id")
    .agg(
        c_index_mean=("c_index", "mean"),
        c_index_std=("c_index", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        ibs_mean=("ibs", "mean"),
        ibs_std=("ibs", "std"),
    )
    .reset_index()
)

summary_per_client.to_csv(OUTPUT_CLIENT_CSV, index=False)

print("Per-client summary (mean ± std over runs):")
print(summary_per_client, "\n")

# ---------------------------------------------------------
# 2) GLOBAL (macro-average per run, excluding centralized_global)
# ---------------------------------------------------------
global_per_run = (
    df_all
    .groupby("run_id")
    .agg(
        c_index=("c_index", "mean"),
        auc=("auc", "mean"),
        ibs=("ibs", "mean"),
    )
    .reset_index()
)

summary_global = (
    global_per_run
    .agg(
        c_index_mean=("c_index", "mean"),
        c_index_std=("c_index", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        ibs_mean=("ibs", "mean"),
        ibs_std=("ibs", "std"),
    )
    .reset_index(drop=True)
)

summary_global.to_csv(OUTPUT_GLOBAL_CSV, index=False)

print("Global summary (macro-average over runs, mean ± std):")
print(summary_global)

print("\nSaved files:")
print(" -", OUTPUT_CLIENT_CSV)
print(" -", OUTPUT_GLOBAL_CSV)