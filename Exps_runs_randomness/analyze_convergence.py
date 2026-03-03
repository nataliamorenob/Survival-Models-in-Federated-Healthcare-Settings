# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # ==============================
# # 1. LOAD ALL RUN FILES
# # ==============================

# base_path = "/scratch/project_2015651/Masters_thesis/results_randomness_exps"

# all_dfs = []

# for i in range(1, 7):
#     file_path = os.path.join(base_path, f"run_{i}.csv")
#     df = pd.read_csv(file_path)
#     df["run_id"] = i
#     all_dfs.append(df)

# # Merge all runs
# df_all = pd.concat(all_dfs, ignore_index=True)

# print("Loaded shape:", df_all.shape)

# # ==============================
# # 2. GLOBAL MEAN PER ROUND
# # ==============================

# round_summary = (
#     df_all.groupby("round")
#     .agg(
#         c_index_mean=("c_index", "mean"),
#         c_index_std=("c_index", "std"),
#         auc_mean=("auc", "mean"),
#         auc_std=("auc", "std"),
#         ibs_mean=("ibs", "mean"),
#         ibs_std=("ibs", "std"),
#     )
#     .reset_index()
# )

# # ==============================
# # 3. ROUND-TO-ROUND DELTA
# # ==============================

# round_summary["delta_c_index"] = round_summary["c_index_mean"].diff()
# round_summary["delta_auc"] = round_summary["auc_mean"].diff()
# round_summary["delta_ibs"] = round_summary["ibs_mean"].diff()

# print("\nRound summary:")
# print(round_summary)

# # ==============================
# # 4. CONVERGENCE DETECTION
# # ==============================

# threshold = 0.002  # You can adjust this

# stable_rounds = round_summary[
#     round_summary["delta_c_index"].abs() < threshold
# ]

# print("\nRounds where improvement < threshold:")
# print(stable_rounds[["round", "delta_c_index"]])

# # ==============================
# # 5. VISUALIZATION
# # ==============================

# plt.figure()
# plt.plot(round_summary["round"], round_summary["c_index_mean"])
# plt.xlabel("Round")
# plt.ylabel("Mean C-index")
# plt.title("Global C-index Convergence")
# plt.show()

# plt.figure()
# plt.plot(round_summary["round"], round_summary["auc_mean"])
# plt.xlabel("Round")
# plt.ylabel("Mean AUC")
# plt.title("Global AUC Convergence")
# plt.show()

# plt.figure()
# plt.plot(round_summary["round"], round_summary["ibs_mean"])
# plt.xlabel("Round")
# plt.ylabel("Mean IBS")
# plt.title("Global IBS Convergence")
# plt.show()

# # ==============================
# # 6. OPTIONAL: WITH ERROR BANDS
# # ==============================

# plt.figure()
# plt.plot(round_summary["round"], round_summary["c_index_mean"])
# plt.fill_between(
#     round_summary["round"],
#     round_summary["c_index_mean"] - round_summary["c_index_std"],
#     round_summary["c_index_mean"] + round_summary["c_index_std"],
#     alpha=0.2
# )
# plt.xlabel("Round")
# plt.ylabel("Mean C-index ± Std")
# plt.title("Global C-index with Variance")
# plt.show()




#Remote
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import t

# ==============================
# 1. LOAD ALL RUN FILES
# ==============================

base_path = "/scratch/project_2015651/Masters_thesis/results_randomness_exps"

all_dfs = []

for i in range(1, 7):
    file_path = os.path.join(base_path, f"run_{i}.csv")
    df = pd.read_csv(file_path)
    df["run_id"] = i
    all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)

print("Loaded shape (raw):", df_all.shape)

# ==============================
# 2. AGGREGATE CLIENTS WITHIN EACH RUN
# ==============================

# We average across the 3 clients inside each run and round
# This gives ONE global C-index per (run, round)

df_clean = (
    df_all
    .groupby(["run_id", "round"], as_index=False)
    .agg({"c_index": "mean"})
)

print("Shape after aggregating clients:", df_clean.shape)

# Number of independent runs
N = df_clean["run_id"].nunique()
df_degrees = N - 1

# t critical value for two-sided 95% CI
t_critical = t.ppf(0.975, df=df_degrees)

print("Number of runs:", N)
print("Degrees of freedom:", df_degrees)
print("Using t critical value:", t_critical)

# ==============================
# 3. GLOBAL MEAN PER ROUND (for plotting)
# ==============================

round_summary = (
    df_clean.groupby("round")
    .agg(
        c_index_mean=("c_index", "mean"),
        c_index_std=("c_index", "std"),
    )
    .reset_index()
)

round_summary["se"] = round_summary["c_index_std"] / np.sqrt(N)

# ==============================
# 4. PAIRED DIFFERENCE BETWEEN CONSECUTIVE ROUNDS
# ==============================

# Create pivot table: rows = round, columns = run_id
pivot_df = df_clean.pivot(index="round", columns="run_id", values="c_index")

results = []

rounds = sorted(pivot_df.index)

for i in range(1, len(rounds)):

    r_prev = rounds[i - 1]
    r_curr = rounds[i]

    # Paired differences per run
    diffs = pivot_df.loc[r_curr] - pivot_df.loc[r_prev]

    mean_diff = diffs.mean()
    std_diff = diffs.std(ddof=1)
    se_diff = std_diff / np.sqrt(N)

    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff

    not_significant = (ci_lower <= 0) and (ci_upper >= 0)

    results.append({
        "round": r_curr,
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "not_significant": not_significant
    })

diff_summary = pd.DataFrame(results)

# ==============================
# 5. REQUIRE STABILITY OVER K ROUNDS
# ==============================

k = 3  # require 3 consecutive non-significant improvements

diff_summary["stable_window"] = (
    diff_summary["not_significant"]
    .rolling(window=k)
    .sum() == k
)

convergence_round = diff_summary.loc[
    diff_summary["stable_window"] == True, "round"
].min()

print("\nConvergence round:", convergence_round)

# ==============================
# 6. OPTIONAL: PLOT MEAN + 95% CI
# ==============================

plt.figure()
plt.plot(round_summary["round"], round_summary["c_index_mean"], label="Mean C-index")

plt.fill_between(
    round_summary["round"],
    round_summary["c_index_mean"] - t_critical * round_summary["se"],
    round_summary["c_index_mean"] + t_critical * round_summary["se"],
    alpha=0.2,
    label="95% CI"
)

plt.xlabel("Round")
plt.ylabel("Mean C-index")
plt.title("C-index with 95% t-Confidence Intervals")
plt.legend()
plt.show()




#LOCAL WITH IMPROVEMENTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import t

base_path = "/scratch/project_2015651/Masters_thesis/results_randomness_exps"

all_dfs = []
for i in range(1, 7):
    file_path = os.path.join(base_path, f"run_{i}.csv")
    df = pd.read_csv(file_path)
    df["run_id"] = i
    all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)

print("Loaded shape (raw):", df_all.shape)
print(f"Unique clients: {df_all['client_id'].nunique()}")
print(f"Unique rounds: {df_all['round'].nunique()}")

# ==============================
# OPTION A: Global Average (your current approach)
# ==============================
df_global = (
    df_all
    .groupby(["run_id", "round"], as_index=False)
    .agg({"c_index": "mean"})
    .rename(columns={"c_index": "c_index_global_avg"})
)

# ==============================
# OPTION B: Worst-case client (fairness)
# ==============================
df_worst = (
    df_all
    .groupby(["run_id", "round"], as_index=False)
    .agg({"c_index": "min"})
    .rename(columns={"c_index": "c_index_worst"})
)

# ==============================
# OPTION C: Variance across clients (heterogeneity)
# ==============================
df_variance = (
    df_all
    .groupby(["run_id", "round"], as_index=False)
    .agg({"c_index": "std"})
    .rename(columns={"c_index": "c_index_std_across_clients"})
)

# Merge all metrics
df_clean = df_global.merge(df_worst, on=["run_id", "round"])
df_clean = df_clean.merge(df_variance, on=["run_id", "round"])

print("\nCombined metrics shape:", df_clean.shape)

N = df_clean["run_id"].nunique()
df_degrees = N - 1
t_critical = t.ppf(0.975, df=df_degrees)

print(f"\nN runs: {N}, df: {df_degrees}, t_critical: {t_critical:.3f}")

# ==============================
# CONVERGENCE ANALYSIS
# ==============================

def analyze_convergence(df_clean, metric_col, N, t_critical, k=3, min_improvement=0.001):
    """
    Analyze convergence for a given metric.
    
    Args:
        df_clean: DataFrame with run_id, round, and metric
        metric_col: Column name to analyze
        N: Number of runs
        t_critical: Critical t-value
        k: Number of consecutive stable rounds required
        min_improvement: Minimum meaningful improvement threshold
    """
    pivot_df = df_clean.pivot(index="round", columns="run_id", values=metric_col)
    
    results = []
    rounds = sorted(pivot_df.index)
    
    for i in range(1, len(rounds)):
        r_prev = rounds[i - 1]
        r_curr = rounds[i]
        
        # Paired differences
        diffs = pivot_df.loc[r_curr] - pivot_df.loc[r_prev]
        
        mean_diff = diffs.mean()
        std_diff = diffs.std(ddof=1)
        se_diff = std_diff / np.sqrt(N)
        
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        # Convergence criteria:
        # 1. Upper bound of CI should be close to zero (not improving much)
        # 2. Lower bound should not be too negative (not degrading)
        converged = (ci_upper < min_improvement) and (ci_lower > -0.01)
        
        # Also flag if significantly improving
        improving = ci_lower > min_improvement
        
        results.append({
            "round": r_curr,
            "mean_diff": mean_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "converged": converged,
            "still_improving": improving
        })
    
    diff_summary = pd.DataFrame(results)
    
    # Find k consecutive converged rounds
    diff_summary["stable_window"] = (
        diff_summary["converged"]
        .rolling(window=k)
        .sum() == k
    )
    
    convergence_round = diff_summary.loc[
        diff_summary["stable_window"] == True, "round"
    ].min()
    
    return diff_summary, convergence_round

# Analyze all three metrics
print("\n" + "="*60)
print("CONVERGENCE ANALYSIS")
print("="*60)

metrics_to_analyze = [
    ("c_index_global_avg", "Global Average C-index"),
    ("c_index_worst", "Worst-case Client C-index"),
    ("c_index_std_across_clients", "Std Dev across Clients")
]

for metric_col, metric_name in metrics_to_analyze:
    print(f"\n{metric_name}:")
    print("-" * 60)
    
    diff_summary, conv_round = analyze_convergence(
        df_clean, metric_col, N, t_critical, k=3
    )
    
    print(diff_summary[["round", "mean_diff", "ci_lower", "ci_upper", 
                        "converged", "still_improving"]])
    
    if pd.notna(conv_round):
        print(f"\n✓ Converged at round: {conv_round}")
    else:
        print("\n✗ No convergence detected (still improving or unstable)")

# ==============================
# VISUALIZATION
# ==============================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Global average with CI
round_summary = df_clean.groupby("round").agg({
    "c_index_global_avg": ["mean", "std"]
}).reset_index()
round_summary.columns = ["round", "mean", "std"]
round_summary["se"] = round_summary["std"] / np.sqrt(N)

axes[0, 0].plot(round_summary["round"], round_summary["mean"], 
                label="Mean C-index", linewidth=2)
axes[0, 0].fill_between(
    round_summary["round"],
    round_summary["mean"] - t_critical * round_summary["se"],
    round_summary["mean"] + t_critical * round_summary["se"],
    alpha=0.2,
    label="95% CI"
)
axes[0, 0].set_xlabel("Round")
axes[0, 0].set_ylabel("Global Avg C-index")
axes[0, 0].set_title("Global Average Performance")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Worst-case client
worst_summary = df_clean.groupby("round").agg({
    "c_index_worst": ["mean", "std"]
}).reset_index()
worst_summary.columns = ["round", "mean", "std"]
worst_summary["se"] = worst_summary["std"] / np.sqrt(N)

axes[0, 1].plot(worst_summary["round"], worst_summary["mean"], 
                label="Worst Client", linewidth=2, color='red')
axes[0, 1].fill_between(
    worst_summary["round"],
    worst_summary["mean"] - t_critical * worst_summary["se"],
    worst_summary["mean"] + t_critical * worst_summary["se"],
    alpha=0.2,
    color='red',
    label="95% CI"
)
axes[0, 1].set_xlabel("Round")
axes[0, 1].set_ylabel("Worst Client C-index")
axes[0, 1].set_title("Worst-Case Client (Fairness)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Heterogeneity (variance across clients)
var_summary = df_clean.groupby("round").agg({
    "c_index_std_across_clients": ["mean", "std"]
}).reset_index()
var_summary.columns = ["round", "mean", "std"]
var_summary["se"] = var_summary["std"] / np.sqrt(N)

axes[1, 0].plot(var_summary["round"], var_summary["mean"], 
                label="Std Dev", linewidth=2, color='purple')
axes[1, 0].fill_between(
    var_summary["round"],
    var_summary["mean"] - t_critical * var_summary["se"],
    var_summary["mean"] + t_critical * var_summary["se"],
    alpha=0.2,
    color='purple',
    label="95% CI"
)
axes[1, 0].set_xlabel("Round")
axes[1, 0].set_ylabel("Std Dev of C-index Across Clients")
axes[1, 0].set_title("Client Heterogeneity")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Per-client trajectories (sample from one run)
sample_run = df_all[df_all["run_id"] == 1]
for client_id in sample_run["client_id"].unique():
    client_data = sample_run[sample_run["client_id"] == client_id]
    axes[1, 1].plot(client_data["round"], client_data["c_index"], 
                    marker='o', label=f"Client {client_id}", alpha=0.7)

axes[1, 1].set_xlabel("Round")
axes[1, 1].set_ylabel("C-index")
axes[1, 1].set_title("Individual Client Trajectories (Run 1)")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_path, "convergence_analysis_comprehensive.png"), 
            dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("Analysis complete! Plot saved.")