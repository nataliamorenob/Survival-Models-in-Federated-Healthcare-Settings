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