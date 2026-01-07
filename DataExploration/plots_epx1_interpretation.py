# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # IMPORTANT: adjust this import if your dataset path is different
# from FedTcgaBrca import FedTcgaBrca


# def load_center_data(center_id):
#     """
#     Load train + test data for one center
#     Returns a DataFrame with columns: event, time
#     """
#     dfs = []

#     for is_train in [True, False]:
#         dataset = FedTcgaBrca(center=center_id, train=is_train)

#         X_list, y_list = [], []
#         for X, y in dataset:
#             X_list.append(X.numpy())
#             y_list.append(y.numpy())

#         df_y = pd.DataFrame(y_list, columns=["event", "time"])
#         dfs.append(df_y)

#     return pd.concat(dfs, ignore_index=True)


# def main():
#     # Load C0 and C1
#     df_c0 = load_center_data(center_id=0)
#     df_c1 = load_center_data(center_id=1)

#     # Keep EVENTS ONLY (delta = 1)
#     df_c0_evt = df_c0[df_c0["event"] == 1]
#     df_c1_evt = df_c1[df_c1["event"] == 1]

#     print(f"C0 events: {len(df_c0_evt)}")
#     print(f"C1 events: {len(df_c1_evt)}")

#     # Shared binning for fair comparison
#     t_min = min(df_c0_evt["time"].min(), df_c1_evt["time"].min())
#     t_max = max(df_c0_evt["time"].max(), df_c1_evt["time"].max())
#     bins = np.linspace(t_min, t_max, 30)

#     # Plot
#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)
#     sns.histplot(
#         df_c0_evt["time"],
#         bins=bins,
#         kde=True,
#         color="tab:blue",
#         alpha=0.7
#     )
#     plt.title(f"Center 0 – Event times only (n={len(df_c0_evt)})")
#     plt.xlabel("Time")
#     plt.ylabel("Count")
#     plt.grid(alpha=0.3)

#     plt.subplot(1, 2, 2)
#     sns.histplot(
#         df_c1_evt["time"],
#         bins=bins,
#         kde=True,
#         color="tab:orange",
#         alpha=0.7
#     )
#     plt.title(f"Center 1 – Event times only (n={len(df_c1_evt)})")
#     plt.xlabel("Time")
#     plt.ylabel("Count")
#     plt.grid(alpha=0.3)

#     plt.suptitle("Histogram of Event Times (δ = 1): C0 vs C1", fontsize=14, weight="bold")
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()


# if __name__ == "__main__":
#     main()

import matplotlib
matplotlib.use("Agg")  # REQUIRED on Puhti / headless systems

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from flamby.datasets.fed_tcga_brca import FedTcgaBrca


def load_center_data(center_id):
    dfs = []
    for is_train in [True, False]:
        dataset = FedTcgaBrca(center=center_id, train=is_train)

        y_list = []
        for _, y in dataset:
            y_list.append(y.numpy())

        df_y = pd.DataFrame(y_list, columns=["event", "time"])
        dfs.append(df_y)

    return pd.concat(dfs, ignore_index=True)


def main():
    df_c0 = load_center_data(center_id=0)
    df_c1 = load_center_data(center_id=1)

    df_c0_evt = df_c0[df_c0["event"] == 1]
    df_c1_evt = df_c1[df_c1["event"] == 1]

    print(f"C0 events: {len(df_c0_evt)}")
    print(f"C1 events: {len(df_c1_evt)}")

    t_min = min(df_c0_evt["time"].min(), df_c1_evt["time"].min())
    t_max = max(df_c0_evt["time"].max(), df_c1_evt["time"].max())
    bins = np.linspace(t_min, t_max, 30)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df_c0_evt["time"], bins=bins, kde=True, alpha=0.7)
    plt.title(f"Center 0 – Event times only (n={len(df_c0_evt)})")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    sns.histplot(df_c1_evt["time"], bins=bins, kde=True, alpha=0.7)
    plt.title(f"Center 1 – Event times only (n={len(df_c1_evt)})")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)

    plt.suptitle("Histogram of Event Times (δ = 1): C0 vs C1", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # SAVE instead of show
    output_path = "event_times_c0_vs_c1.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
