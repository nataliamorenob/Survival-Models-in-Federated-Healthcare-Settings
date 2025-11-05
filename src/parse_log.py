import re
import pandas as pd

# === Read your log ===
log_path = "/scratch/project_2015651/Masters_thesis/src/results/20251105_160651/experiment_20251105_160651.log"  
with open(log_path, "r") as f:
    log = f.read()

# === Regex patterns ===
client_pattern = re.compile(
    r"Round (\d+).*?→ (\d+) \(ID (\d+)\): C-index=([0-9.nan]+), AUC=([0-9.]+), IBS=([0-9.]+)",
    re.DOTALL,
)
agg_pattern = re.compile(
    r"Round (\d+) - Aggregated Metrics → C-index=([0-9.]+), AUC=([0-9.]+), IBS=([0-9.]+)"
)

# === Extract client metrics ===
client_rows = []
for match in client_pattern.findall(log):
    round_num, client_name, client_id, c_index, auc, ibs = match
    client_rows.append({
        "Round": int(round_num),
        "Client": int(client_name),
        "C_index": float(c_index) if c_index != "nan" else None,
        "AUC": float(auc),
        "IBS": float(ibs),
    })

clients_df = pd.DataFrame(client_rows)

# === Extract aggregated metrics ===
agg_rows = []
for match in agg_pattern.findall(log):
    round_num, c_index, auc, ibs = match
    agg_rows.append({
        "Round": int(round_num),
        "C_index": float(c_index),
        "AUC": float(auc),
        "IBS": float(ibs),
    })

agg_df = pd.DataFrame(agg_rows)

# === Best metrics per client ===
best_client_metrics = []
for client, group in clients_df.groupby("Client"):
    best_cindex = group.loc[group["C_index"].idxmax()] if group["C_index"].notna().any() else None
    best_auc = group.loc[group["AUC"].idxmax()]
    best_ibs = group.loc[group["IBS"].idxmin()]

    best_client_metrics.append({
        "Client": client,
        "Best_Cindex_Round": int(best_cindex["Round"]) if best_cindex is not None else None,
        "Best_Cindex": best_cindex["C_index"] if best_cindex is not None else None,
        "Best_AUC_Round": int(best_auc["Round"]),
        "Best_AUC": best_auc["AUC"],
        "Best_IBS_Round": int(best_ibs["Round"]),
        "Best_IBS": best_ibs["IBS"],
    })

best_clients_df = pd.DataFrame(best_client_metrics).sort_values("Client")

# === Best aggregated metrics ===
best_agg = {
    "Best_Cindex_Round": int(agg_df.loc[agg_df["C_index"].idxmax(), "Round"]),
    "Best_Cindex": agg_df["C_index"].max(),
    "Best_AUC_Round": int(agg_df.loc[agg_df["AUC"].idxmax(), "Round"]),
    "Best_AUC": agg_df["AUC"].max(),
    "Best_IBS_Round": int(agg_df.loc[agg_df["IBS"].idxmin(), "Round"]),
    "Best_IBS": agg_df["IBS"].min(),
}

# === Display results ===
print("\n=== Best Metrics per Client ===")
print(best_clients_df.to_string(index=False))

print("\n=== Best Aggregated (Server) Metrics ===")
for k, v in best_agg.items():
    print(f"{k}: {v}")
