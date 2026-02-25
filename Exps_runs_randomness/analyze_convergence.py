import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# 1. LOAD ALL RUN FILES
# ==============================

base_path = "/Users/nataliamorenoblasco/Desktop/Master_Thesis/Exps_runs_randomness"

all_dfs = []

for i in range(1, 7):
    file_path = os.path.join(base_path, f"run_{i}.csv")
    df = pd.read_csv(file_path)
    df["run_id"] = i
    all_dfs.append(df)

# Merge all runs
df_all = pd.concat(all_dfs, ignore_index=True)

print("Loaded shape:", df_all.shape)

# ==============================
# 2. GLOBAL MEAN PER ROUND
# ==============================

round_summary = (
    df_all.groupby("round")
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

# ==============================
# 3. ROUND-TO-ROUND DELTA
# ==============================

round_summary["delta_c_index"] = round_summary["c_index_mean"].diff()
round_summary["delta_auc"] = round_summary["auc_mean"].diff()
round_summary["delta_ibs"] = round_summary["ibs_mean"].diff()

print("\nRound summary:")
print(round_summary)

# ==============================
# 4. CONVERGENCE DETECTION
# ==============================

threshold = 0.002  # You can adjust this

stable_rounds = round_summary[
    round_summary["delta_c_index"].abs() < threshold
]

print("\nRounds where improvement < threshold:")
print(stable_rounds[["round", "delta_c_index"]])

# ==============================
# 5. VISUALIZATION
# ==============================

plt.figure()
plt.plot(round_summary["round"], round_summary["c_index_mean"])
plt.xlabel("Round")
plt.ylabel("Mean C-index")
plt.title("Global C-index Convergence")
plt.show()

plt.figure()
plt.plot(round_summary["round"], round_summary["auc_mean"])
plt.xlabel("Round")
plt.ylabel("Mean AUC")
plt.title("Global AUC Convergence")
plt.show()

plt.figure()
plt.plot(round_summary["round"], round_summary["ibs_mean"])
plt.xlabel("Round")
plt.ylabel("Mean IBS")
plt.title("Global IBS Convergence")
plt.show()

# ==============================
# 6. OPTIONAL: WITH ERROR BANDS
# ==============================

plt.figure()
plt.plot(round_summary["round"], round_summary["c_index_mean"])
plt.fill_between(
    round_summary["round"],
    round_summary["c_index_mean"] - round_summary["c_index_std"],
    round_summary["c_index_mean"] + round_summary["c_index_std"],
    alpha=0.2
)
plt.xlabel("Round")
plt.ylabel("Mean C-index ± Std")
plt.title("Global C-index with Variance")
plt.show()