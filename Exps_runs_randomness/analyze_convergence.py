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
# """
# Federated Learning Convergence Analysis

# This script analyzes convergence patterns in federated learning experiments using
# statistically rigorous methods appropriate for FL settings.

# Methodology justification and citations: see CONVERGENCE_ANALYSIS_JUSTIFICATION.md

# Key methodological choices:
# - Paired t-tests for round-to-round comparisons (Box et al., 2005)
# - Practical significance threshold of 0.5% (Sullivan & Feinn, 2012)
# - k=3 consecutive stable rounds requirement (Prechelt, 1998)
# - Multiple metrics: global, worst-case, heterogeneity (Li et al., 2019; Zhao et al., 2018)
# - Oscillation patterns as convergence signal (Khaled et al., 2020)

# For complete citations, see: references.bib
# """

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from scipy.stats import t

# # ==============================
# # 1. LOAD ALL RUN FILES
# # ==============================

# base_path = "/scratch/project_2015651/Masters_thesis/results_randomness_exps"

# all_dfs = []

for i in range(1, 11):
    file_path = os.path.join(base_path, f"run_{i}.csv")
    df = pd.read_csv(file_path)
    df["run_id"] = i
    all_dfs.append(df)

# df_all = pd.concat(all_dfs, ignore_index=True)

# print("Loaded shape (raw):", df_all.shape)

# # ==============================
# # 2. AGGREGATE CLIENTS WITHIN EACH RUN
# # ==============================

# # We average across the 3 clients inside each run and round
# # This gives ONE global C-index per (run, round)

# df_clean = (
#     df_all
#     .groupby(["run_id", "round"], as_index=False)
#     .agg({"c_index": "mean"})
# )

# print("Shape after aggregating clients:", df_clean.shape)

# # Number of independent runs
# N = df_clean["run_id"].nunique()
# df_degrees = N - 1

# # t critical value for two-sided 95% CI
# t_critical = t.ppf(0.975, df=df_degrees)

# print("Number of runs:", N)
# print("Degrees of freedom:", df_degrees)
# print("Using t critical value:", t_critical)

# # ==============================
# # 3. GLOBAL MEAN PER ROUND (for plotting)
# # ==============================

# round_summary = (
#     df_clean.groupby("round")
#     .agg(
#         c_index_mean=("c_index", "mean"),
#         c_index_std=("c_index", "std"),
#     )
#     .reset_index()
# )

# round_summary["se"] = round_summary["c_index_std"] / np.sqrt(N)

# # ==============================
# # 4. PAIRED DIFFERENCE BETWEEN CONSECUTIVE ROUNDS
# # ==============================

# # Create pivot table: rows = round, columns = run_id
# pivot_df = df_clean.pivot(index="round", columns="run_id", values="c_index")

# results = []

# rounds = sorted(pivot_df.index)

# for i in range(1, len(rounds)):

#     r_prev = rounds[i - 1]
#     r_curr = rounds[i]

#     # Paired differences per run
#     diffs = pivot_df.loc[r_curr] - pivot_df.loc[r_prev]

#     mean_diff = diffs.mean()
#     std_diff = diffs.std(ddof=1)
#     se_diff = std_diff / np.sqrt(N)

#     ci_lower = mean_diff - t_critical * se_diff
#     ci_upper = mean_diff + t_critical * se_diff

#     not_significant = (ci_lower <= 0) and (ci_upper >= 0)

#     results.append({
#         "round": r_curr,
#         "mean_diff": mean_diff,
#         "ci_lower": ci_lower,
#         "ci_upper": ci_upper,
#         "not_significant": not_significant
#     })

# diff_summary = pd.DataFrame(results)

# # ==============================
# # 5. REQUIRE STABILITY OVER K ROUNDS
# # ==============================

# k = 3  # require 3 consecutive non-significant improvements

# diff_summary["stable_window"] = (
#     diff_summary["not_significant"]
#     .rolling(window=k)
#     .sum() == k
# )

# convergence_round = diff_summary.loc[
#     diff_summary["stable_window"] == True, "round"
# ].min()

# print("\nConvergence round:", convergence_round)

# # ==============================
# # 6. OPTIONAL: PLOT MEAN + 95% CI
# # ==============================

# plt.figure()
# plt.plot(round_summary["round"], round_summary["c_index_mean"], label="Mean C-index")

# plt.fill_between(
#     round_summary["round"],
#     round_summary["c_index_mean"] - t_critical * round_summary["se"],
#     round_summary["c_index_mean"] + t_critical * round_summary["se"],
#     alpha=0.2,
#     label="95% CI"
# )

# plt.xlabel("Round")
# plt.ylabel("Mean C-index")
# plt.title("C-index with 95% t-Confidence Intervals")
# plt.legend()
# plt.show()



















#LOCAL WITH IMPROVEMENTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import t

base_path = "/scratch/project_2015651/Masters_thesis/results_randomness_exps"

all_dfs = []
for i in range(1, 11):
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

def analyze_convergence(df_clean, metric_col, N, t_critical, k=3, 
                        practical_threshold=0.005, max_degradation=0.01, max_cv=0.15):
    """
    Two-stage convergence analysis: temporal stability + stochastic stability.
    
    Stage 1 (Temporal): Detects when round-to-round improvements become negligible
    Stage 2 (Stochastic): Validates reproducibility across random seeds
    
    Args:
        df_clean: DataFrame with run_id, round, and metric
        metric_col: Column name to analyze
        N: Number of runs
        t_critical: Critical t-value
        k: Number of consecutive stable rounds required
        practical_threshold: Practical significance threshold (e.g., 0.5% improvement)
        max_degradation: Maximum acceptable degradation (e.g., 1%)
        max_cv: Maximum coefficient of variation for stochastic stability (default: 0.15 = 15%)
        
    Citations:
        - Temporal: Box et al. (2005), Sullivan & Feinn (2012), Prechelt (1998)
        - Stochastic: Henderson et al. (2018), Bouthillier et al. (2019)
        - CV threshold: Reed et al. (2002) - CV<15% indicates low variability
    """
    pivot_df = df_clean.pivot(index="round", columns="run_id", values=metric_col)
    
    results = []
    rounds = sorted(pivot_df.index)
    
    for i in range(1, len(rounds)):
        r_prev = rounds[i - 1]
        r_curr = rounds[i]
        
        # Paired differences for temporal convergence
        diffs = pivot_df.loc[r_curr] - pivot_df.loc[r_prev]
        
        mean_diff = diffs.mean()
        std_diff = diffs.std(ddof=1)
        se_diff = std_diff / np.sqrt(N)
        
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        # STAGE 1: TEMPORAL CONVERGENCE
        # Mean change is small AND not significantly improving
        practical_converged = (abs(mean_diff) < practical_threshold) and (ci_lower < practical_threshold)
        not_degrading = ci_lower > -max_degradation
        temporal_converged = practical_converged and not_degrading
        
        # Flag significant improvement
        significantly_improving = ci_lower > practical_threshold
        
        # Flag oscillation (CI crosses zero with small magnitude)
        oscillating = (ci_lower < 0) and (ci_upper > 0) and (abs(mean_diff) < practical_threshold)
        
        # STAGE 2: STOCHASTIC STABILITY
        # Calculate coefficient of variation (CV) across runs at current round
        values_curr = pivot_df.loc[r_curr].values
        mean_val = values_curr.mean()
        std_val = values_curr.std(ddof=1)
        cv = abs(std_val / mean_val) if abs(mean_val) > 1e-6 else np.inf
        
        # Stochastic stability: CV below threshold
        stochastic_stable = cv < max_cv
        
        # COMBINED CRITERION: Both temporal AND stochastic
        fully_converged = temporal_converged and stochastic_stable
        
        results.append({
            "round": r_curr,
            "mean_diff": mean_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "cv": cv,
            "temporal_converged": temporal_converged,
            "stochastic_stable": stochastic_stable,
            "converged": fully_converged,  # Combined criterion
            "significantly_improving": significantly_improving,
            "oscillating": oscillating
        })
    
    diff_summary = pd.DataFrame(results)
    
    # STAGE 1: Find k consecutive temporally converged rounds
    diff_summary["temporal_stable_window"] = (
        diff_summary["temporal_converged"]
        .rolling(window=k)
        .sum() == k
    )
    
    temporal_convergence_round = diff_summary.loc[
        diff_summary["temporal_stable_window"] == True, "round"
    ].min()
    
    # STAGE 2: Find k consecutive fully converged rounds (temporal + stochastic)
    diff_summary["optimal_stable_window"] = (
        diff_summary["converged"]
        .rolling(window=k)
        .sum() == k
    )
    
    optimal_convergence_round = diff_summary.loc[
        diff_summary["optimal_stable_window"] == True, "round"
    ].min()
    
    # Alternative: Find when oscillation becomes stable
    diff_summary["oscillation_window"] = (
        diff_summary["oscillating"]
        .rolling(window=k)
        .sum() == k
    )
    
    oscillation_convergence = diff_summary.loc[
        diff_summary["oscillation_window"] == True, "round"
    ].min()
    
    return diff_summary, temporal_convergence_round, optimal_convergence_round, oscillation_convergence

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
    print("-" * 80)
    
    diff_summary, temporal_conv_round, optimal_conv_round, osc_round = analyze_convergence(
        df_clean, metric_col, N, t_critical, k=3, 
        practical_threshold=0.005, max_degradation=0.01
    )
    
    # Show key columns
    display_cols = ["round", "mean_diff", "ci_lower", "ci_upper", "cv",
                    "temporal_converged", "stochastic_stable", "converged", 
                    "significantly_improving", "oscillating"]
    print(diff_summary[display_cols].to_string())
    
    print("\n" + "-" * 80)
    
    # STAGE 1: Temporal convergence
    if pd.notna(temporal_conv_round):
        print(f"✓ TEMPORAL CONVERGENCE (Stage 1) at round: {temporal_conv_round}")
        print(f"  (3 consecutive rounds with |change| < 0.5% and not significantly improving)")
    else:
        print("✗ No temporal convergence detected")
    
    # STAGE 2: Optimal convergence (temporal + stochastic)
    if pd.notna(optimal_conv_round):
        print(f"✓ OPTIMAL CONVERGENCE (Stage 2) at round: {optimal_conv_round}")
        print(f"  (3 consecutive rounds with temporal stability AND CV < 15%)")
    else:
        print("✗ No optimal convergence detected (high variance persists)")
    
    if pd.notna(osc_round):
        print(f"✓ STABLE OSCILLATION at round: {osc_round}")
        print(f"  (3 consecutive rounds oscillating around zero with |change| < 0.5%)")
    else:
        print("✗ No stable oscillation pattern detected")
    
    # Summary statistics for post-round-10 behavior
    post_10 = diff_summary[diff_summary["round"] >= 10]
    if len(post_10) > 0:
        print(f"\nPost-round-10 behavior:")
        print(f"  - Mean absolute change: {post_10['mean_diff'].abs().mean():.6f}")
        print(f"  - Max absolute change: {post_10['mean_diff'].abs().max():.6f}")
        print(f"  - Mean CV: {post_10['cv'].mean():.4f} (Std: {post_10['cv'].std():.4f})")
        print(f"  - Rounds with |change| < 0.005: {(post_10['mean_diff'].abs() < 0.005).sum()}/{len(post_10)}")
        print(f"  - Rounds with CV < 0.15: {(post_10['cv'] < 0.15).sum()}/{len(post_10)}")
        print(f"  - Rounds oscillating: {post_10['oscillating'].sum()}/{len(post_10)}")

# ==============================
# VISUALIZATION
# ==============================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Global average with CI and convergence markers
round_summary = df_clean.groupby("round").agg({
    "c_index_global_avg": ["mean", "std"]
}).reset_index()
round_summary.columns = ["round", "mean", "std"]
round_summary["se"] = round_summary["std"] / np.sqrt(N)

# Get convergence info
diff_global, temporal_conv_global, optimal_conv_global, osc_global = analyze_convergence(
    df_clean, "c_index_global_avg", N, t_critical, k=3
)

axes[0, 0].plot(round_summary["round"], round_summary["mean"], 
                label="Mean C-index", linewidth=2, marker='o', markersize=4)
axes[0, 0].fill_between(
    round_summary["round"],
    round_summary["mean"] - t_critical * round_summary["se"],
    round_summary["mean"] + t_critical * round_summary["se"],
    alpha=0.2,
    label="95% CI"
)

# Mark convergence regions
if pd.notna(temporal_conv_global):
    axes[0, 0].axvline(x=temporal_conv_global, color='blue', linestyle=':', 
                       linewidth=2, label=f'Temporal conv (round {temporal_conv_global})')
if pd.notna(optimal_conv_global):
    axes[0, 0].axvline(x=optimal_conv_global, color='green', linestyle='--', 
                       linewidth=2, label=f'Optimal conv (round {optimal_conv_global})')
    axes[0, 0].axvspan(optimal_conv_global, round_summary["round"].max(), 
                       alpha=0.1, color='green')
elif pd.notna(osc_global):
    axes[0, 0].axvline(x=osc_global, color='orange', linestyle='--', 
                       linewidth=2, label=f'Stable oscillation (round {osc_global})')
    axes[0, 0].axvspan(osc_global, round_summary["round"].max(), 
                       alpha=0.1, color='orange')

axes[0, 0].set_xlabel("Round", fontsize=12)
axes[0, 0].set_ylabel("Global Avg C-index", fontsize=12)
axes[0, 0].set_title("Global Average Performance", fontsize=14, fontweight='bold')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Round-to-round changes with convergence threshold
axes[0, 1].plot(diff_global["round"], diff_global["mean_diff"], 
                marker='o', linewidth=2, label="Mean change")
axes[0, 1].fill_between(
    diff_global["round"],
    diff_global["ci_lower"],
    diff_global["ci_upper"],
    alpha=0.2,
    label="95% CI"
)
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0, 1].axhline(y=0.005, color='red', linestyle='--', linewidth=1, 
                   alpha=0.5, label='Practical threshold (±0.5%)')
axes[0, 1].axhline(y=-0.005, color='red', linestyle='--', linewidth=1, alpha=0.5)

# Highlight converged rounds
converged_rounds = diff_global[diff_global["converged"]]["round"]
if len(converged_rounds) > 0:
    axes[0, 1].scatter(converged_rounds, 
                       diff_global[diff_global["converged"]]["mean_diff"],
                       color='green', s=100, marker='*', zorder=5,
                       label='Converged rounds')

axes[0, 1].set_xlabel("Round", fontsize=12)
axes[0, 1].set_ylabel("Change in C-index", fontsize=12)
axes[0, 1].set_title("Round-to-Round Improvements", fontsize=14, fontweight='bold')
axes[0, 1].legend(loc='upper right')
axes[0, 1].grid(True, alpha=0.3)

# Plot 2: Worst-case client
worst_summary = df_clean.groupby("round").agg({
    "c_index_worst": ["mean", "std"]
}).reset_index()
worst_summary.columns = ["round", "mean", "std"]
worst_summary["se"] = worst_summary["std"] / np.sqrt(N)

diff_worst, temporal_conv_worst, optimal_conv_worst, osc_worst = analyze_convergence(
    df_clean, "c_index_worst", N, t_critical, k=3
)

axes[1, 0].plot(worst_summary["round"], worst_summary["mean"], 
                label="Worst Client", linewidth=2, color='red', marker='o', markersize=4)
axes[1, 0].fill_between(
    worst_summary["round"],
    worst_summary["mean"] - t_critical * worst_summary["se"],
    worst_summary["mean"] + t_critical * worst_summary["se"],
    alpha=0.2,
    color='red',
    label="95% CI"
)

# Mark convergence
if pd.notna(temporal_conv_worst):
    axes[1, 0].axvline(x=temporal_conv_worst, color='blue', linestyle=':', 
                       linewidth=2, label=f'Temporal conv (round {temporal_conv_worst})')
if pd.notna(optimal_conv_worst):
    axes[1, 0].axvline(x=optimal_conv_worst, color='green', linestyle='--', 
                       linewidth=2, label=f'Optimal conv (round {optimal_conv_worst})')
    axes[1, 0].axvspan(optimal_conv_worst, worst_summary["round"].max(), 
                       alpha=0.1, color='green')

axes[1, 0].set_xlabel("Round", fontsize=12)
axes[1, 0].set_ylabel("Worst Client C-index", fontsize=12)
axes[1, 0].set_title("Worst-Case Client (Fairness)", fontsize=14, fontweight='bold')
axes[1, 0].legend(loc='lower right')
axes[1, 0].grid(True, alpha=0.3)

# Plot 3: Heterogeneity (variance across clients)
var_summary = df_clean.groupby("round").agg({
    "c_index_std_across_clients": ["mean", "std"]
}).reset_index()
var_summary.columns = ["round", "mean", "std"]
var_summary["se"] = var_summary["std"] / np.sqrt(N)

diff_var, temporal_conv_var, optimal_conv_var, osc_var = analyze_convergence(
    df_clean, "c_index_std_across_clients", N, t_critical, k=3
)

axes[1, 1].plot(var_summary["round"], var_summary["mean"], 
                label="Std Dev", linewidth=2, color='purple', marker='o', markersize=4)
axes[1, 1].fill_between(
    var_summary["round"],
    var_summary["mean"] - t_critical * var_summary["se"],
    var_summary["mean"] + t_critical * var_summary["se"],
    alpha=0.2,
    color='purple',
    label="95% CI"
)

# Mark if heterogeneity stabilizes
if pd.notna(temporal_conv_var):
    axes[1, 1].axvline(x=temporal_conv_var, color='blue', linestyle=':', 
                       linewidth=2, label=f'Temporal conv (round {temporal_conv_var})')
if pd.notna(optimal_conv_var):
    axes[1, 1].axvline(x=optimal_conv_var, color='green', linestyle='--', 
                       linewidth=2, label=f'Optimal conv (round {optimal_conv_var})')
elif pd.notna(osc_var):
    axes[1, 1].axvline(x=osc_var, color='orange', linestyle='--', 
                       linewidth=2, label=f'Stable (round {osc_var})')

axes[1, 1].set_xlabel("Round", fontsize=12)
axes[1, 1].set_ylabel("Std Dev of C-index Across Clients", fontsize=12)
axes[1, 1].set_title("Client Heterogeneity Over Time", fontsize=14, fontweight='bold')
axes[1, 1].legend(loc='upper right')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_path, "convergence_analysis_comprehensive.png"), 
            dpi=300, bbox_inches='tight')
print("\n" + "="*80)
print(f"Analysis complete! Plot saved to: {os.path.join(base_path, 'convergence_analysis_comprehensive.png')}")
plt.show()

# ==============================
# ADDITIONAL PLOT: Per-client trajectories from ALL runs
# ==============================
fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot all runs with transparency
for run in df_all["run_id"].unique():
    run_data = df_all[df_all["run_id"] == run]
    for client_id in run_data["client_id"].unique():
        client_data = run_data[run_data["client_id"] == client_id]
        ax.plot(client_data["round"], client_data["c_index"], 
                alpha=0.3, linewidth=1, color=f'C{client_id}')

# Plot mean trajectories per client (across all runs)
for client_id in df_all["client_id"].unique():
    client_mean = df_all[df_all["client_id"] == client_id].groupby("round")["c_index"].mean()
    ax.plot(client_mean.index, client_mean.values, 
            marker='o', label=f"Client {client_id} (mean)", 
            linewidth=2.5, color=f'C{client_id}')

# Mark convergence region if detected
if pd.notna(temporal_conv_global):
    ax.axvline(x=temporal_conv_global, color='blue', linestyle=':', 
               linewidth=2, alpha=0.7, label=f'Temporal convergence')
if pd.notna(optimal_conv_global):
    ax.axvline(x=optimal_conv_global, color='green', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Optimal convergence')
    ax.axvspan(optimal_conv_global, df_all["round"].max(), 
               alpha=0.05, color='green')

ax.set_xlabel("Round", fontsize=12)
ax.set_ylabel("C-index", fontsize=12)
ax.set_title("Individual Client Trajectories (All Runs)", fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_path, "per_client_trajectories.png"), 
            dpi=300, bbox_inches='tight')
print(f"Per-client trajectories saved to: {os.path.join(base_path, 'per_client_trajectories.png')}")
plt.show()

# ==============================
# FINAL SUMMARY
# ==============================
print("\n" + "="*80)
print("CONVERGENCE SUMMARY")
print("="*80)

summary_data = []

# Re-run analyses to collect summary
for metric_col, metric_name in metrics_to_analyze:
    diff_summary, temporal_conv_round, optimal_conv_round, osc_round = analyze_convergence(
        df_clean, metric_col, N, t_critical, k=3
    )
    
    # Post-round-10 statistics
    post_10 = diff_summary[diff_summary["round"] >= 10]
    mean_abs_change = post_10['mean_diff'].abs().mean() if len(post_10) > 0 else np.nan
    max_abs_change = post_10['mean_diff'].abs().max() if len(post_10) > 0 else np.nan
    mean_cv = post_10['cv'].mean() if len(post_10) > 0 else np.nan
    pct_small_changes = (post_10['mean_diff'].abs() < 0.005).mean() * 100 if len(post_10) > 0 else 0
    pct_low_cv = (post_10['cv'] < 0.15).mean() * 100 if len(post_10) > 0 else 0
    
    summary_data.append({
        'Metric': metric_name.replace('C-index', '').strip(),
        'Temporal Conv': int(temporal_conv_round) if pd.notna(temporal_conv_round) else 'Not detected',
        'Optimal Conv': int(optimal_conv_round) if pd.notna(optimal_conv_round) else 'Not detected',
        'Oscillation': int(osc_round) if pd.notna(osc_round) else 'Not detected',
        'Mean |Δ| (post-10)': f'{mean_abs_change:.6f}' if not np.isnan(mean_abs_change) else 'N/A',
        'Mean CV (post-10)': f'{mean_cv:.4f}' if not np.isnan(mean_cv) else 'N/A',
        '% |Δ|<0.5%': f'{pct_small_changes:.1f}%',
        '% CV<15%': f'{pct_low_cv:.1f}%'
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("TWO-STAGE CONVERGENCE CRITERION:")
print("\n• Stage 1 - Temporal Convergence: First round where training stabilizes")
print("  (3 consecutive rounds with |change| < 0.5% and not significantly improving)")
print("  → Indicates when improvements have plateaued")
print("\n• Stage 2 - Optimal Convergence: First round with both temporal stability AND")
print("  low variance (CV < 15% across runs)")
print("  → Recommended round for reporting results (stable + reproducible)")
print("\n• Oscillation: Model oscillates around zero for 3 consecutive rounds")
print("  (typical late-stage FL behavior)")
print("\n• Mean CV (post-10): Average coefficient of variation (std/mean) after round 10")
print("  CV < 15% indicates good reproducibility (Reed et al., 2002)")
print("\n" + "="*80)
print("\nREFERENCES:")
print("- Henderson et al. (2018): Deep RL That Matters (variance in RL)")
print("- Bouthillier et al. (2019): Accounting for Variance in ML Experiments")
print("- Reed et al. (2002): CV < 15% threshold in experimental sciences")
print("="*80)

# Save summary to CSV
summary_df.to_csv(os.path.join(base_path, "convergence_summary.csv"), index=False)
print(f"\nSummary saved to: {os.path.join(base_path, 'convergence_summary.csv')}")
print("="*80)