# from flamby.datasets.fed_tcga_brca import FedTcgaBrca
# from torch.utils.data import DataLoader, ConcatDataset
# import numpy as np
# import logging


# class DatasetManager:
#     def __init__(self, config, client_idx):
#         """Initialize the DatasetManager with the given configuration."""
#         self.config = config
#         self.client_idx = client_idx

#     def get_federated_dataloaders(self):
#         """
#         Create dataloaders for federated training.
#         Splits original training data into 80% train and 20% val,
#         ensures validation has at least one event (E == 1),
#         and scales the first feature (age).
#         """
#         # logger = logging.getLogger("main")
#         # logging.basicConfig(level=logging.INFO)
#         logger = logging.getLogger("main")

#         # also print directly, since Flower subprocesses might swallow logs
#         def log_and_print(msg, level="info"):
#             if level == "info":
#                 logger.info(msg)
#             elif level == "warning":
#                 logger.warning(msg)
#             elif level == "error":
#                 logger.error(msg)
#             print(msg, flush=True)
#         dataloaders = {}

#         for center in self.config.centers:
#             # Load FLamby datasets
#             full_train_ds = FedTcgaBrca(center=center, train=True)
#             test_ds = FedTcgaBrca(center=center, train=False)
#             print(f"[DEBUG] Center {center}: train={len(full_train_ds)} test={len(test_ds)}")
            
#             # 1- Compute scaling parameters for the "age" feature
#             train_ages = []
#             for sample in full_train_ds:
#                 if isinstance(sample, dict):  # newer FLamby format
#                     age_val = sample["features"][0].item()
#                 else:  # older FLamby format: (x, (t, e))
#                     x, _ = sample
#                     age_val = x[0].item()
#                 train_ages.append(age_val)

#             age_mean = np.mean(train_ages)
#             age_std = np.std(train_ages)

#             # 2- Split the training dataset into 80/20
#             full_train_list = list(full_train_ds)
#             np.random.shuffle(full_train_list)
#             split_idx = int(len(full_train_list) * 0.8)
#             train_ds_split = full_train_list[:split_idx]
#             val_ds_split = full_train_list[split_idx:]

#             # 3- Ensure validation set has at least one event (E == 1)
#             def get_event(sample):
#                 if isinstance(sample, dict):
#                     return sample["event"]
#                 else:
#                     _, y = sample
#                     return int(y[1].item()) if hasattr(y[1], "item") else int(y[1])

#             has_event_in_val = any(get_event(s) == 1 for s in val_ds_split)
#             if not has_event_in_val:
#                 event_indices = [i for i, s in enumerate(train_ds_split) if get_event(s) == 1]
#                 if event_indices:
#                     moved_sample = train_ds_split.pop(event_indices[0])
#                     val_ds_split.append(moved_sample)
#                     logger.warning(f"[Center {center}] Added 1 event sample from train → val to ensure event presence.")
#                 else:
#                     logger.warning(f"[Center {center}] No events available in train to move to validation!")

#             # Ensure there is at least one event (E == 1) in train, val, and test
#             def ensure_event_presence(dataset, dataset_name):
#                 has_event = any(get_event(s) == 1 for s in dataset)
#                 if not has_event:
#                     logger.warning(f"[{dataset_name}] No events found in the dataset!")

#             ensure_event_presence(train_ds_split, f"Train (Center {center})")
#             ensure_event_presence(val_ds_split, f"Validation (Center {center})")
#             ensure_event_presence(test_ds, f"Test (Center {center})")

#             # 4- Process datasets: normalize and extract numpy arrays
#             def _process_dataset(dataset):
#                 features, time, event = [], [], []
#                 for sample in dataset:
#                     if isinstance(sample, dict):  # new FLamby
#                         x = sample["features"].clone().detach().numpy()
#                         t = sample["time"]
#                         e = sample["event"]
#                     else:  # old FLamby
#                         x, y = sample
#                         x = x.clone().detach().numpy()
#                         y_np = y.clone().detach().numpy()
#                         t, e = y_np[0], y_np[1]

#                     # Normalize age feature
#                     if age_std > 0:
#                         x[0] = (x[0] - age_mean) / age_std
#                     else:
#                         x[0] = 0.0

#                     features.append(x[:39])
#                     time.append(float(t))
#                     event.append(int(e))

#                 return features, time, event

#             # 5- Apply processing to splits
#             train_features, train_time, train_event = _process_dataset(train_ds_split)
#             val_features, val_time, val_event = _process_dataset(val_ds_split)
#             test_features, test_time, test_event = _process_dataset(test_ds)

#             msg = (
#                 f"[Center {center}] Split summary → "
#                 f"Train: {len(train_features)} samples ({sum(train_event)} events), "
#                 f"Val: {len(val_features)} samples ({sum(val_event)} events), "
#                 f"Test: {len(test_features)} samples ({sum(test_event)} events)"
#             )
#             log_and_print(msg)


#             dataloaders[center] = {
#                 "train": {"features": train_features, "time": train_time, "event": train_event},
#                 "val": {"features": val_features, "time": val_time, "event": val_event},
#                 "test": {"features": test_features, "time": test_time, "event": test_event},
#             }

#         return dataloaders

#     def get_local_dataloaders(self):
#         """Create dataloaders for local training (single center)."""
#         center = self.config.centers[0]
#         train_ds = FedTcgaBrca(center=center, train=True)
#         test_ds = FedTcgaBrca(center=center, train=False)
#         return {
#             "train": DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True),
#             "test": DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False),
#         }

#     def get_centralized_dataloaders(self):
#         """Create dataloaders for centralized training (all centers combined)."""
#         train_datasets = [FedTcgaBrca(center=c, train=True) for c in self.config.centers]
#         test_datasets = [FedTcgaBrca(center=c, train=False) for c in self.config.centers]

#         train_ds = ConcatDataset(train_datasets)
#         test_ds = ConcatDataset(test_datasets)

#         return {
#             "train": DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True),
#             "test": DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False),
#         }

#     def get_dataloaders(self):
#         """Return the appropriate dataloaders based on the training mode."""
#         if self.config.training_mode == "federated":
#             return self.get_federated_dataloaders()
#         elif self.config.training_mode == "local":
#             return self.get_local_dataloaders()
#         elif self.config.training_mode == "centralized":
#             return self.get_centralized_dataloaders()
#         else:
#             raise ValueError(f"Unsupported training mode: {self.config.training_mode}")


from flamby.datasets.fed_tcga_brca import FedTcgaBrca
from torch.utils.data import ConcatDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os

class DatasetManager:
    def __init__(self, config, client_idx):
        """Initialize DatasetManager with configuration and client index."""
        self.config = config
        self.client_idx = client_idx
        self.logger = logging.getLogger("main")


        # Replace print statements with logger calls
        def log_and_print(msg, level="info"):
            if level == "info":
                self.logger.info(msg)
            elif level == "warning":
                self.logger.warning(msg)
            elif level == "error":
                self.logger.error(msg)
            elif level == "debug":
                self.logger.debug(msg)
            # Print to stdout for immediate visibility
            print(msg, flush=True)

        self.log_and_print = log_and_print

    # ----------------------------------------------------------------------
    # Main interface
    # ----------------------------------------------------------------------
    def get_dataloaders(self):
        """Return dataloaders depending on training mode."""
        mode = self.config.training_mode
        self.log_and_print(f"[DatasetManager] Building dataloaders for mode: {mode}")

        if mode == "federated":
            return self.get_federated_dataloaders()
        elif mode == "local":
            return self.get_local_dataloaders()
        elif mode == "centralized":
            return self.get_centralized_dataloaders()
        else:
            raise ValueError(f"Unsupported training mode: {mode}")

    # ----------------------------------------------------------------------
    # FEDERATED LEARNING MODE:
    # ----------------------------------------------------------------------
    def get_federated_dataloaders(self):
        """Prepare per-center train/val/test DataFrames for Federated Learning."""
        dataloaders = {}
        center = self.config.centers[self.client_idx]
        self.log_and_print(f"[Federated] Preparing dataset for center: {center}")

        cache_path = os.path.join(self.config.cache_dir, f"center_{center}_data.pkl")

        # If cached version exists, load it instead of rebuilding
        if os.path.exists(cache_path):
            self.log_and_print(f"[Federated] Using cached dataset for center {center} → {cache_path}")
            dataloaders = pd.read_pickle(cache_path)
            return dataloaders

        # Otherwise, load Fed-TCGA-BRCA data:
        self.log_and_print(f"[Federated] Loading Fed-TCGA-BRCA center {center}...")
        train_ds = FedTcgaBrca(center=center, train=True)
        test_ds = FedTcgaBrca(center=center, train=False)
        self.log_and_print(f"[Center {center}] train={len(train_ds)}, test={len(test_ds)}", "info")

        # Convert to DataFrames:
        df_train = self._to_dataframe(train_ds)
        df_test = self._to_dataframe(test_ds)
        self.log_and_print(
            f"[Center {center}] DataFrames built — train={df_train.shape}, test={df_test.shape}", "info"
        )
        self._log_data_preview(df_train, df_test, center)

        # Normalize age (feature_0):
        if "feature_0" in df_train.columns:
            age_mean = df_train["feature_0"].mean()
            age_std = df_train["feature_0"].std()
            if age_std == 0:
                age_std = 1.0

            for df in [df_train, df_test]:
                df["feature_0"] = (df["feature_0"] - age_mean) / age_std

            self.log_and_print(
                f"[Center {center}] Normalized feature_0 (age): mean={age_mean:.2f}, std={age_std:.2f}"
            )

        # MDEL-SPECIFIC HANDELING:------------------------>
        
        # CoxPH model
        if self.config.model.lower() == "coxph":
            self.log_and_print(f"[Center {center}] Model=CoxPH → Using train/test directly.")
            dataloaders[center] = {"train": df_train, "val": None, "test": df_test}

        # Stacked Logistic Regression (SLR) model
        elif self.config.model.lower() == "slr":
            self.log_and_print(f"[Center {center}] Model=SLR → Performing stacking transformation.")

            # First, find the global max time across all centers for consistent binning
            global_max_time = self._get_global_max_time()

            df_train_stacked = self._stack_data(
                df_train, center=center, split="train", global_max_time=global_max_time
            )
            df_test_stacked = self._stack_data(
                df_test, center=center, split="test", global_max_time=global_max_time
            )

            self.log_and_print(
                f"[Center {center}] After stacking: train={df_train_stacked.shape}, test={df_test_stacked.shape}"
            )

            dataloaders[center] = {"train": df_train_stacked, "val": None, "test": df_test_stacked}
        
        elif self.config.model.lower() in ["rsf", "deepsurv"]:
            
            # NO VAL SET (OLD):
            # # RSF always keeps original DataFrames
            # dataloaders[center] = {
            #     "train_df": df_train,
            #     "test_df": df_test,
            #     "val": None
            # }



            # Event-aware train/validation split (FedSurF++)
            df_train_split, df_val = self._split_train_val_event_aware(df_train, center)

            dataloaders[center] = {
                "train_df": df_train_split,
                "val_df": df_val,
                "test_df": df_test,
            }



            # Extract features
            feature_cols = [c for c in df_train.columns if c.startswith("feature_")]
            X_train = df_train_split[feature_cols].values
            X_val   = df_val[feature_cols].values
            X_test = df_test[feature_cols].values

            # Build structured survival arrays
            from sksurv.util import Surv
            y_train = Surv.from_arrays(
                event=df_train_split["event"].astype(bool),
                time=df_train_split["time"].astype(float)
            )

            y_val = Surv.from_arrays(
                event=df_val["event"].astype(bool),
                time=df_val["time"].astype(float),
            )

            y_test = Surv.from_arrays(
                event=df_test["event"].astype(bool),
                time=df_test["time"].astype(float)
            )

            dataloaders[center]["X_train"] = X_train
            dataloaders[center]["X_test"] = X_test
            dataloaders[center]["X_val"] = X_val
            dataloaders[center]["y_val"] = y_val
            dataloaders[center]["y_train"] = y_train
            dataloaders[center]["y_test"] = y_test

            # ---------------------------------------------
            # Compute client-specific evaluation times
            # ---------------------------------------------
            # event_times = df_test.loc[df_test["event"] == 1, "time"].values

            # if len(event_times) > 1:
            #     eval_times = np.quantile(event_times, np.linspace(0.1, 0.9, 20))
            # else:
            #     mint, maxt = df_test["time"].min(), df_test["time"].max()
            #     eval_times = np.linspace(mint, maxt, 20)

            # eval_times = np.unique(eval_times)
            # dataloaders[center]["eval_times"] = eval_times



            # NEW CORRECTION BC C-INDEX LOW --> Use TRAINING event times, not test events:
            train_event_times = df_train_split.loc[df_train_split["event"] == 1, "time"].values

            if len(train_event_times) > 1:
                # Use quantiles of TRAINING events — avoids early/late flat survival
                eval_times = np.quantile(train_event_times, np.linspace(0.05, 0.95, 50))
            else:
                # Fallback to the training range
                mint, maxt = df_train["time"].min(), df_train["time"].max()
                eval_times = np.linspace(mint, maxt, 50)

            eval_times = np.unique(eval_times)
            dataloaders[center]["eval_times"] = eval_times


            # also store globally in the config
            if not hasattr(self.config, "eval_times_per_client"):
                self.config.eval_times_per_client = {}
            self.config.eval_times_per_client[self.client_idx] = eval_times.tolist()
            
            # NEW --> In order to create a global grid
            if self.config.eval_grid_mode == "global":
                if not hasattr(self.config, "all_train_event_times"):
                    self.config.all_train_event_times = []
                self.config.all_train_event_times.extend(train_event_times.tolist())
                

            self.log_and_print(
                f"[Center {center}] RSF eval_times stored: {np.round(eval_times, 2)}"
            )

    
        
        # Deep models 
        else:
            self.log_and_print(
                f"[Center {center}] Splitting training data into 80% train / 20% val ..."
            )
            df_train_split, df_val = self._split_train_val(df_train, center)
            df_val["feature_0"] = (df_val["feature_0"] - age_mean) / age_std
            dataloaders[center] = {"train": df_train_split, "val": df_val, "test": df_test}

        #  # DEBUG: Save processed data to CSV for inspection
        # try:
        #     # Define output directory (already created by init_logging)
        #     output_dir = self.config.experiment_dir

        #     # Determine which DataFrames exist for this center
        #     data_dict = dataloaders[center]

        #     for split_name, df_split in data_dict.items():
        #         if df_split is not None:
        #             csv_path = os.path.join(
        #                 output_dir, f"center_{center}_{split_name}.csv"
        #             )
        #             df_split.to_csv(csv_path, index=False)
        #             self.log_and_print(f"[Center {center}] Saved {split_name} data to {csv_path}")
        # except Exception as e:
        #     self.log_and_print(f"[Center {center}] Failed to save debug CSVs: {e}", "warning")
            
        pd.to_pickle(dataloaders, cache_path)
        self.log_and_print(f"[Federated] Cached dataset saved for center {center} → {cache_path}")
        self.log_and_print(f"[Federated] Center {center} processed successfully")


        return dataloaders



    # LOCAL TRAINING MODE (single center)
    def get_local_dataloaders(self):
        center = self.config.centers[0]
        self.log_and_print(f"[Local] Loading center {center}...")

        train_ds = FedTcgaBrca(center=center, train=True)
        test_ds = FedTcgaBrca(center=center, train=False)
        df_train = self._to_dataframe(train_ds)
        df_test = self._to_dataframe(test_ds)

        # Normalize age
        age_mean = df_train["feature_0"].mean()
        age_std = df_train["feature_0"].std()
        if age_std == 0:
            age_std = 1.0
        for df in [df_train, df_test]:
            df["feature_0"] = (df["feature_0"] - age_mean) / age_std

        self.log_and_print(
            f"[Local] Normalized feature_0 (age) with mean={age_mean:.2f}, std={age_std:.2f}"
        )

        # if self.config.model.lower() == "coxph":
        #     self.log_and_print("[Local] Model=CoxPH → Using train/test directly.")
        #     return {"train": df_train, "val": None, "test": df_test}

        # if self.config.model.lower() == "slr":
        #     self.log_and_print("[Local] Model=SLR → Performing stacking transformation.")

        #     global_max_time = self._get_global_max_time()

        #     df_train_stacked = self._stack_data(
        #         df_train, center=center, split="train", global_max_time=global_max_time
        #     )
        #     df_test_stacked = self._stack_data(
        #         df_test, center=center, split="test", global_max_time=global_max_time
        #     )

        #     return {"train": df_train_stacked, "val": None, "test": df_test_stacked}

        if self.config.model.lower() in ["rsf", "deepsurv"]:
            df_train_split, df_val = self._split_train_val_event_aware(df_train, center)

            feature_cols = [c for c in df_train.columns if c.startswith("feature_")]
            X_train = df_train_split[feature_cols].values
            X_val = df_val[feature_cols].values
            X_test = df_test[feature_cols].values

            from sksurv.util import Surv
            y_train = Surv.from_arrays(
                event=df_train_split["event"].astype(bool),
                time=df_train_split["time"].astype(float),
            )
            y_val = Surv.from_arrays(
                event=df_val["event"].astype(bool),
                time=df_val["time"].astype(float),
            )
            y_test = Surv.from_arrays(
                event=df_test["event"].astype(bool),
                time=df_test["time"].astype(float),
            )

            return {
                "train_df": df_train_split,
                "val_df": df_val,
                "test_df": df_test,
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
            }

        self.log_and_print("[Local] Unsupported model for local dataloaders.", "warning")
        return {"train": df_train, "val": None, "test": df_test}


    # CENTRALIZED TRAINING MODE (all centers combined):
    def get_centralized_dataloaders(self):
        self.log_and_print(f"[Centralized] Combining data from centers: {self.config.centers}")

        center_train_dfs = {}
        center_test_dfs = {}

        # Concatenate centers
        for center in self.config.centers:
            train_ds = FedTcgaBrca(center=center, train=True)
            test_ds = FedTcgaBrca(center=center, train=False)
            center_train_dfs[center] = self._to_dataframe(train_ds)
            center_test_dfs[center] = self._to_dataframe(test_ds)

        df_train = pd.concat(list(center_train_dfs.values()), ignore_index=True)
        df_test = pd.concat(list(center_test_dfs.values()), ignore_index=True)

        # Normalize age globally using the combined train set (BEFORE split)
        age_mean = df_train["feature_0"].mean()
        age_std = df_train["feature_0"].std()
        if age_std == 0:
            age_std = 1.0

        # Normalize the concatenated dataframes FIRST
        df_train["feature_0"] = (df_train["feature_0"] - age_mean) / age_std
        df_test["feature_0"] = (df_test["feature_0"] - age_mean) / age_std
        
        # THEN normalize per-center test data (which are separate from df_test)
        # These are the original per-center DataFrames, not slices of df_test
        for center in center_test_dfs.keys():
            center_test_dfs[center] = center_test_dfs[center].copy()
            center_test_dfs[center]["feature_0"] = (center_test_dfs[center]["feature_0"] - age_mean) / age_std

        self.log_and_print(
            f"[Centralized] Normalized feature_0 (age) with mean={age_mean:.2f}, std={age_std:.2f}"
        )

        per_center = {}

        # # CoxPH model
        # if self.config.model.lower() == "coxph":
        #     self.log_and_print("[Centralized] Model=CoxPH → Using train/test directly.")

        #     for center, df_center_test in center_test_dfs.items():
        #         per_center[center] = {"test": df_center_test}

        #     return {
        #         "global": {"train": df_train, "val": None, "test": df_test},
        #         "per_center": per_center,
        #     }

        # # Stacked Logistic Regression (SLR) model
        # if self.config.model.lower() == "slr":
        #     self.log_and_print("[Centralized] Model=SLR → Performing stacking transformation.")

        #     global_max_time = self._get_global_max_time()

        #     df_train_stacked = self._stack_data(
        #         df_train, center="centralized", split="train", global_max_time=global_max_time
        #     )
        #     df_test_stacked = self._stack_data(
        #         df_test, center="centralized", split="test", global_max_time=global_max_time
        #     )

        #     for center, df_center_test in center_test_dfs.items():
        #         per_center[center] = {
        #             "test": self._stack_data(
        #                 df_center_test,
        #                 center=f"centralized_{center}",
        #                 split="test",
        #                 global_max_time=global_max_time,
        #             )
        #         }

        #     return {
        #         "global": {"train": df_train_stacked, "val": None, "test": df_test_stacked},
        #         "per_center": per_center,
        #     }

        # Random Survival Forest (RSF) and DeepSurv models
        if self.config.model.lower() in ["rsf", "deepsurv"]:
            # Validation checks BEFORE split
            self.log_and_print(
                f"[Centralized] Pre-split validation → "
                f"df_train: {len(df_train)} samples, {df_train['event'].sum()} events, "
                f"time range=[{df_train['time'].min():.2f}, {df_train['time'].max():.2f}], "
                f"feature_0 (age) range=[{df_train['feature_0'].min():.2f}, {df_train['feature_0'].max():.2f}]"
            )
            
            df_train_split, df_val = self._split_train_val_event_aware(df_train, center=0)

            self.log_and_print(
                "[Centralized] Combined dataset sizes → "
                f"train={len(df_train_split)} (events={df_train_split['event'].sum()}), "
                f"val={len(df_val)} (events={df_val['event'].sum()}), "
                f"test={len(df_test)} (events={df_test['event'].sum()})"
            )
            
            # Validation: Check for duplicates or data leakage
            train_times = set(zip(df_train_split['time'].values, df_train_split['event'].values))
            test_times = set(zip(df_test['time'].values, df_test['event'].values))
            overlap = train_times.intersection(test_times)
            if len(overlap) > 0:
                self.log_and_print(
                    f"[Centralized] WARNING: {len(overlap)} overlapping (time, event) pairs between train and test!",
                    "warning"
                )
            else:
                self.log_and_print("[Centralized] No train/test overlap detected (good!)")

            global_data = {
                "train_df": df_train_split,
                "val_df": df_val,
                "test_df": df_test,
            }

            feature_cols = [c for c in df_train.columns if c.startswith("feature_")]
            X_train = df_train_split[feature_cols].values
            X_val = df_val[feature_cols].values
            X_test = df_test[feature_cols].values

            from sksurv.util import Surv
            y_train = Surv.from_arrays(
                event=df_train_split["event"].astype(bool),
                time=df_train_split["time"].astype(float),
            )
            y_val = Surv.from_arrays(
                event=df_val["event"].astype(bool),
                time=df_val["time"].astype(float),
            )
            y_test = Surv.from_arrays(
                event=df_test["event"].astype(bool),
                time=df_test["time"].astype(float),
            )

            global_data.update(
                {
                    "X_train": X_train,
                    "X_val": X_val,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_val": y_val,
                    "y_test": y_test,
                }
            )

            for center, df_center_test in center_test_dfs.items():
                X_test_center = df_center_test[feature_cols].values
                y_test_center = Surv.from_arrays(
                    event=df_center_test["event"].astype(bool),
                    time=df_center_test["time"].astype(float),
                )
                per_center[center] = {
                    "X_test": X_test_center,
                    "y_test": y_test_center,
                    "test_df": df_center_test,
                }

            self.log_and_print("[Centralized] Finished building centralized dataloaders")
            return {"global": global_data, "per_center": per_center}

        # If model is not RSF/DeepSurv, raise error
        raise ValueError(f"[Centralized] Unsupported model: {self.config.model}")

    # ----------------------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------------------
    def _to_dataframe(self, dataset):
        """Convert FedTcgaBrca dataset into a pandas DataFrame."""
        X_list, y_event, y_time = [], [], []
        for X, y in dataset:
            X_list.append(X.numpy())
            y_event.append(int(y[0]))  # event
            y_time.append(float(y[1]))  # time

        df = pd.DataFrame(X_list, columns=[f"feature_{i}" for i in range(X_list[0].shape[0])])
        df["event"] = y_event
        df["time"] = y_time
        return df

    # def _split_train_val(self, df_train, center):
    #     """Split training set into train (80%) and val (20%), ensuring at least 1 event in val."""
    #     df_train_split, df_val = train_test_split(
    #         df_train, test_size=0.2, random_state=42, stratify=df_train["event"]
    #     )

    #     # Ensure validation has at least 1 event
    #     if df_val["event"].sum() == 0:
    #         event_rows = df_train_split[df_train_split["event"] == 1]
    #         if not event_rows.empty:
    #             row_to_move = event_rows.sample(n=1, random_state=42)
    #             df_val = pd.concat([df_val, row_to_move], ignore_index=True)
    #             df_train_split = df_train_split.drop(row_to_move.index)
    #             self.log_and_print(f"[Center {center}] ⚠️ Added 1 event sample from train → val to ensure event presence.", "warning")
    #         else:
    #             self.log_and_print(f"[Center {center}] ⚠️ No events available to move to validation!", "warning")

    #     self.log_and_print(
    #         f"[Center {center}] Final splits — Train: {len(df_train_split)}, "
    #         f"Val: {len(df_val)}, Events (val): {df_val['event'].sum()}",
    #         "info",
    #     )
    #     return df_train_split, df_val


    def _split_train_val_event_aware(self, df_train, center):
        """
        Event-aware train/validation split for survival models (RSF / FedSurF++).
        Ensures validation contains at least 2 events.
        """

        n_events = int(df_train["event"].sum())

        if n_events < 3:
            raise ValueError(
                f"[Center {center}] Too few events ({n_events}) to build validation set."
            )

        # Adaptive validation size
        if n_events >= 20:
            val_size = 0.30
        elif n_events >= 10:
            val_size = 0.25
        else:
            val_size = 0.20

        # Ensure at least 2 validation events
        min_val_events = 2
        expected_val_events = int(val_size * n_events)

        if expected_val_events < min_val_events:
            val_size = min_val_events / n_events

        df_tr, df_val = train_test_split(
            df_train,
            test_size=val_size,
            stratify=df_train["event"],
            random_state=self.config.random_state + int(center),
        )

        if df_val["event"].sum() < min_val_events:
            raise RuntimeError(
                f"[Center {center}] Validation split has "
                f"{df_val['event'].sum()} events (< {min_val_events})."
            )

        self.log_and_print(
            f"[Center {center}] Event-aware split → "
            f"train={len(df_tr)} (events={df_tr['event'].sum()}), "
            f"val={len(df_val)} (events={df_val['event'].sum()})"
        )

        return df_tr, df_val




    def _stack_data(self, df, center=None, split="", global_max_time=None):
        """
        Create a stacked dataset using time bins for logistic regression survival modeling.
        Each patient is repeated for each time bin up to their observed time.
        A binary label is 1 if the event occurred in that time bin, else 0.
        The time bin itself is one-hot encoded as a feature.
        """
        self.log_and_print(
            f"[Center {center}] [{split}] Starting stacking with {self.config.num_time_bins} bins... "
            f"input rows={len(df)}",
            "info",
        )

        if df.empty:
            self.log_and_print(f"[Center {center}] [{split}] Input DataFrame is empty, cannot stack.", "warning")
            # Return an empty DataFrame with expected columns
            feature_cols = [c for c in df.columns if c.startswith("feature_")]
            bin_cols = [f"time_bin_{i}" for i in range(self.config.num_time_bins)]
            return pd.DataFrame(columns=feature_cols + bin_cols + ["event"])

        # 1. Define time bins based on the GLOBAL max time
        if global_max_time is None:
            self.log_and_print(
                f"[Center {center}] [{split}] No global_max_time provided, using local max time.", "warning"
            )
            global_max_time = df["time"].max()

        if global_max_time == 0:
            global_max_time = 1 # Avoid empty bins if max time is 0
            
        bins = np.linspace(0, global_max_time, self.config.num_time_bins + 1)
        
        self.log_and_print(f"[Center {center}] [{split}] Global time bins created: {np.round(bins, 2)}", "debug")

        stacked_rows = []
        feature_cols = [c for c in df.columns if c.startswith("feature_")]

        # 2. For each patient, create a row for each time bin they were at risk
        for _, row in df.iterrows():
            patient_time = row["time"]
            patient_event = row["event"]
            features = row[feature_cols].values

            # Find which bin the patient's event time falls into
            event_bin_idx = np.digitize(patient_time, bins) - 1
            
            # A patient is at risk in all bins up to and including their event bin
            for bin_idx in range(event_bin_idx + 1):
                # Label is 1 only if it's the event bin and an event occurred
                label = 1 if (bin_idx == event_bin_idx and patient_event == 1) else 0
                
                # Create a new row with original features, the bin index, and the label
                new_row = np.concatenate([features, [bin_idx, label]])
                stacked_rows.append(new_row)

        if not stacked_rows:
            self.log_and_print(f"[Center {center}] [{split}] No rows were created during stacking.", "warning")
            bin_cols = [f"time_bin_{i}" for i in range(self.config.num_time_bins)]
            return pd.DataFrame(columns=feature_cols + bin_cols + ["event"])

        # 3. Create a DataFrame from the stacked rows
        stacked_df = pd.DataFrame(stacked_rows, columns=feature_cols + ["time_bin", "event"])
        stacked_df["time_bin"] = stacked_df["time_bin"].astype(int)

        # DEBUG: Save the dataset with the categorical 'time_bin' column
        try:
            pre_dummy_path = os.path.join(
                self.config.experiment_dir, f"center_{center}_{split}_pre_dummy.csv"
            )
            stacked_df.to_csv(pre_dummy_path, index=False)
            self.log_and_print(f"[Debug] Saved pre-dummy data to {pre_dummy_path}", "debug")
        except Exception as e:
            self.log_and_print(f"[Debug] Failed to save pre-dummy CSV: {e}", "warning")

        # 4. One-hot encode the 'time_bin' categorical feature
        time_bin_dummies = pd.get_dummies(stacked_df["time_bin"], prefix="time_bin", dtype=int)
        
        # Ensure all possible bin columns are present, even if some bins had no data
        for i in range(self.config.num_time_bins):
            col_name = f"time_bin_{i}"
            if col_name not in time_bin_dummies.columns:
                time_bin_dummies[col_name] = 0
        
        # Order columns correctly
        time_bin_dummies = time_bin_dummies[[f"time_bin_{i}" for i in range(self.config.num_time_bins)]]

        # 5. Combine original features with the new one-hot encoded time features
        final_df = pd.concat([stacked_df.drop(columns=["time_bin"]), time_bin_dummies], axis=1)

        # DEBUG: Save the final dataset after one-hot encoding
        try:
            post_dummy_path = os.path.join(
                self.config.experiment_dir, f"center_{center}_{split}_post_dummy.csv"
            )
            final_df.to_csv(post_dummy_path, index=False)
            self.log_and_print(f"[Debug] Saved post-dummy data to {post_dummy_path}", "debug")
        except Exception as e:
            self.log_and_print(f"[Debug] Failed to save post-dummy CSV: {e}", "warning")

        # Log summary info
        self.log_and_print(f"[Center {center}] [{split}] Finished stacking — output rows={len(final_df)}", "info")
        self.log_and_print(f"[Center {center}] [{split}] Event counts after stacking:\n{final_df['event'].value_counts().to_dict()}")
        self.log_and_print(f"[Center {center}] [{split}] Final columns: {final_df.columns.tolist()}", "debug")
        
        return final_df

    # ----------------------------------------------------------------------
    # GLOBAL MAX TIME (shared across all centers)
    # ----------------------------------------------------------------------
    def _get_global_max_time(self):
        """
        Compute or cache the GLOBAL maximum 'time' across all centers
        to ensure consistent time-bin edges (as in Westers et al. 2025).

        Returns:
            float: Global maximum survival time across all centers.
        """
        # Cache path so we don’t recompute every run
        cache_path = os.path.join(self.config.cache_dir, "global_max_time.txt")

        # If already computed, reuse it
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    global_max_time = float(f.read().strip())
                self.log_and_print(f"[Global] Loaded cached global_max_time={global_max_time:.2f}")
                return global_max_time
            except Exception:
                self.log_and_print("[Global] Failed to read cached global_max_time. Recomputing...", "warning")

        # Otherwise, compute from all centers’ local data
        self.log_and_print("[Global] Computing global maximum survival time across centers...")
        max_times = []
        for c in self.config.centers:
            try:
                ds = FedTcgaBrca(center=c, train=True)
                times = [float(y[1]) for _, y in ds]
                if times:
                    max_times.append(max(times))
                self.log_and_print(f"  Center {c}: local max={max(times):.2f}")
            except Exception as e:
                self.log_and_print(f"  [Center {c}] ⚠️ Failed to read times: {e}", "warning")

        # Determine overall maximum
        global_max_time = float(max(max_times)) if max_times else 1.0
        self.log_and_print(f"[Global] Computed global_max_time={global_max_time:.2f}")

        # Cache to disk for future calls
        os.makedirs(self.config.cache_dir, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(str(global_max_time))

        return global_max_time


    def _log_data_preview(self, df_train, df_test, center):
        """Helper to log basic dataset statistics and a few example rows."""
        try:
            self.log_and_print(f"[Center {center}] ── TRAIN STATS ──", "info")
            self.log_and_print(f"  → Shape: {df_train.shape}")
            self.log_and_print(f"  → Event counts: {df_train['event'].value_counts().to_dict()}")
            self.log_and_print(
                f"  → Time stats: min={df_train['time'].min():.1f}, "
                f"max={df_train['time'].max():.1f}, mean={df_train['time'].mean():.1f}"
            )
            self.log_and_print(f"  → Example rows:\n{df_train.head(3).to_string(index=False)}")

            self.log_and_print(f"[Center {center}] ── TEST STATS ──", "info")
            self.log_and_print(f"  → Shape: {df_test.shape}")
            self.log_and_print(f"  → Event counts: {df_test['event'].value_counts().to_dict()}")
            self.log_and_print(
                f"  → Time stats: min={df_test['time'].min():.1f}, "
                f"max={df_test['time'].max():.1f}, mean={df_test['time'].mean():.1f}"
            )
            self.log_and_print(f"  → Example rows:\n{df_test.head(3).to_string(index=False)}")
        except Exception as e:
            self.log_and_print(f"[Center {center}] ⚠️ Failed to log data preview: {e}", "warning")

    
