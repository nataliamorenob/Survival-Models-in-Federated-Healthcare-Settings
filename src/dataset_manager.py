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

        # Model-specific handling:
        
        # CoxPH model
        if self.config.model.lower() == "coxph":
            self.log_and_print(f"[Center {center}] Model=CoxPH → Using train/test directly.")
            dataloaders[center] = {"train": df_train, "val": None, "test": df_test}

        # Stacked Logistic Regression (SLR) model
        elif self.config.model.lower() == "slr":
            self.log_and_print(f"[Center {center}] Model=SLR → Performing stacking transformation.")
            df_train_stacked = self._stack_data(df_train, center=center, split="train")
            df_test_stacked = self._stack_data(df_test, center=center, split="test")

            self.log_and_print(
                f"[Center {center}] After stacking: train={df_train_stacked.shape}, test={df_test_stacked.shape}"
            )

            dataloaders[center] = {"train": df_train_stacked, "val": None, "test": df_test_stacked}

        # Deep models 
        else:
            self.log_and_print(
                f"[Center {center}] Splitting training data into 80% train / 20% val ..."
            )
            df_train_split, df_val = self._split_train_val(df_train, center)
            df_val["feature_0"] = (df_val["feature_0"] - age_mean) / age_std
            dataloaders[center] = {"train": df_train_split, "val": df_val, "test": df_test}

         # DEBUG: Save processed data to CSV for inspection
        try:
            # Define output directory (already created by init_logging)
            output_dir = self.config.experiment_dir

            # Determine which DataFrames exist for this center
            data_dict = dataloaders[center]

            for split_name, df_split in data_dict.items():
                if df_split is not None:
                    csv_path = os.path.join(
                        output_dir, f"center_{center}_{split_name}.csv"
                    )
                    df_split.to_csv(csv_path, index=False)
                    self.log_and_print(f"[Center {center}] Saved {split_name} data to {csv_path}")
        except Exception as e:
            self.log_and_print(f"[Center {center}] Failed to save debug CSVs: {e}", "warning")
            
        pd.to_pickle(dataloaders, cache_path)
        self.log_and_print(f"[Federated] Cached dataset saved for center {center} → {cache_path}")


        self.log_and_print(f"[Federated] Center {center} processed successfully")
        return dataloaders


    # ----------------------------------------------------------------------
    # LOCAL TRAINING MODE (single center)
    # ----------------------------------------------------------------------
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

        self.log_and_print(f"[Local] Normalized age (feature_0) with mean={age_mean:.2f}, std={age_std:.2f}")
        return {"train": df_train, "val": None, "test": df_test}

    # ----------------------------------------------------------------------
    # CENTRALIZED TRAINING MODE (all centers combined)
    # ----------------------------------------------------------------------
    def get_centralized_dataloaders(self):
        self.log_and_print(f"[Centralized] Combining data from centers: {self.config.centers}")
        train_datasets = [FedTcgaBrca(center=c, train=True) for c in self.config.centers]
        test_datasets = [FedTcgaBrca(center=c, train=False) for c in self.config.centers]

        df_train = pd.concat([self._to_dataframe(ds) for ds in train_datasets], ignore_index=True)
        df_test = pd.concat([self._to_dataframe(ds) for ds in test_datasets], ignore_index=True)

        # Normalize age globally
        age_mean = df_train["feature_0"].mean()
        age_std = df_train["feature_0"].std()
        if age_std == 0:
            age_std = 1.0
        for df in [df_train, df_test]:
            df["feature_0"] = (df["feature_0"] - age_mean) / age_std

        self.log_and_print(f"[Centralized] Normalized age (feature_0) with mean={age_mean:.2f}, std={age_std:.2f}")

        if self.config.model.lower() == "coxph":
            self.log_and_print(f"[Centralized] Model=CoxPH → Using existing train/test only.")
            return {"train": df_train, "val": None, "test": df_test}

        self.log_and_print(f"[Centralized] Splitting 80/20 train/val ...")
        df_train_split, df_val = self._split_train_val(df_train, "centralized")
        df_val["feature_0"] = (df_val["feature_0"] - age_mean) / age_std

        self.log_and_print(f"[Centralized] Finished building centralized dataloaders ✅")
        return {"train": df_train_split, "val": df_val, "test": df_test}

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

    def _split_train_val(self, df_train, center):
        """Split training set into train (80%) and val (20%), ensuring at least 1 event in val."""
        df_train_split, df_val = train_test_split(
            df_train, test_size=0.2, random_state=42, stratify=df_train["event"]
        )

        # Ensure validation has at least 1 event
        if df_val["event"].sum() == 0:
            event_rows = df_train_split[df_train_split["event"] == 1]
            if not event_rows.empty:
                row_to_move = event_rows.sample(n=1, random_state=42)
                df_val = pd.concat([df_val, row_to_move], ignore_index=True)
                df_train_split = df_train_split.drop(row_to_move.index)
                self.log_and_print(f"[Center {center}] ⚠️ Added 1 event sample from train → val to ensure event presence.", "warning")
            else:
                self.log_and_print(f"[Center {center}] ⚠️ No events available to move to validation!", "warning")

        self.log_and_print(
            f"[Center {center}] Final splits — Train: {len(df_train_split)}, "
            f"Val: {len(df_val)}, Events (val): {df_val['event'].sum()}",
            "info",
        )
        return df_train_split, df_val

    def _stack_data(self, df, center=None, split=""):
        """
        Create a stacked dataset for logistic regression survival modeling.
        Each patient is repeated for each time up to their observed time.
        Binary label = 1 if event occurred at that time, else 0.
        """
        self.log_and_print(f"[Center {center}] [{split}] Starting stacking... input rows={len(df)}", "info")

        stacked_rows = []
        for _, row in df.iterrows():
            time = int(row["time"])
            event = int(row["event"])
            features = row[[c for c in df.columns if c.startswith("feature_")]].values
            for t in range(1, time + 1):
                label = 1 if (t == time and event == 1) else 0
                stacked_rows.append(np.concatenate([features, [label]]))

        stacked_df = pd.DataFrame(
            stacked_rows,
            columns=[c for c in df.columns if c.startswith("feature_")] + ["event"]
        )

        # Log summary info
        self.log_and_print(f"[Center {center}] [{split}] Finished stacking — output rows={len(stacked_df)}", "info")
        self.log_and_print(f"[Center {center}] [{split}] Event counts after stacking:\n{stacked_df['event'].value_counts().to_dict()}")
        self.log_and_print(f"[Center {center}] [{split}] Example rows:\n{stacked_df.head(5).to_string(index=False)}", "debug")

        return stacked_df

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
