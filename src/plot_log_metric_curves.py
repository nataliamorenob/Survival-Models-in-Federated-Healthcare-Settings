import re
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# === CONFIG ===
# Path to your log file
log_path = Path("/scratch/project_2015651/Masters_thesis/src/results/20251105_160651/experiment_20251105_160651.log")

# Output directory for the figure
output_dir = log_path.parent
output_path = output_dir / "federated_metrics.png"

# === REGEX PATTERNS ===
pattern_round = re.compile(r"round\s+(\d+)", re.IGNORECASE)
pattern_metrics = re.compile(r"C-index=([0-9\.nanNaN]+), AUC=([0-9\.nanNaN]+), IBS=([0-9\.nanNaN]+)")

# === PARSE LOG ===
rounds, cindex, auc, ibs = [], [], [], []
current_round = None

with open(log_path, "r") as f:
    for line in f:
        # Detect round number
        round_match = pattern_round.search(line)
        if round_match:
            current_round = int(round_match.group(1))
            continue

        # Detect aggregated or global metrics
        if "[Server]" in line or "[Global]" in line or "aggregated" in line.lower():
            metrics_match = pattern_metrics.search(line)
            if metrics_match and current_round is not None:
                try:
                    c = float(metrics_match.group(1)) if "nan" not in metrics_match.group(1).lower() else None
                    a = float(metrics_match.group(2)) if "nan" not in metrics_match.group(2).lower() else None
                    i = float(metrics_match.group(3)) if "nan" not in metrics_match.group(3).lower() else None
                    rounds.append(current_round)
                    cindex.append(c)
                    auc.append(a)
                    ibs.append(i)
                except ValueError:
                    continue

# === BUILD DATAFRAME ===
df = pd.DataFrame({
    "Round": rounds,
    "C-index": cindex,
    "AUC": auc,
    "IBS": ibs
}).dropna().sort_values("Round")

if df.empty:
    print("No metrics found in the log file.")
else:
    print(f"Parsed {len(df)} metric entries from {log_path.name}")
    print(df.head())

    # === PLOT ===
    plt.figure(figsize=(10, 6))
    plt.plot(df["Round"], df["C-index"], label="C-index", marker="o")
    plt.plot(df["Round"], df["AUC"], label="AUC", marker="s")
    plt.plot(df["Round"], df["IBS"], label="IBS (lower is better)", marker="^")
    plt.xlabel("Federated Round")
    plt.ylabel("Metric Value")
    plt.title("Federated Training Metrics over Rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # === SAVE FIGURE ===
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {output_path}")
