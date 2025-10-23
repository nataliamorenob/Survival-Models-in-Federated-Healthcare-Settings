from flamby.datasets.fed_tcga_brca import FedTcgaBrca
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import logging


class DatasetManager:
    def __init__(self, config, client_idx):
        """Initialize the DatasetManager with the given configuration."""
        self.config = config
        self.client_idx = client_idx

    def get_federated_dataloaders(self):
        """
        Create dataloaders for federated training.
        Splits original training data into 80% train and 20% val,
        ensures validation has at least one event (E == 1),
        and scales the first feature (age).
        """
        # logger = logging.getLogger("main")
        # logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("main")

        # also print directly, since Flower subprocesses might swallow logs
        def log_and_print(msg, level="info"):
            if level == "info":
                logger.info(msg)
            elif level == "warning":
                logger.warning(msg)
            elif level == "error":
                logger.error(msg)
            print(msg, flush=True)
        dataloaders = {}

        for center in self.config.centers:
            # Load FLamby datasets
            full_train_ds = FedTcgaBrca(center=center, train=True)
            test_ds = FedTcgaBrca(center=center, train=False)

            # 1- Compute scaling parameters for the "age" feature
            train_ages = []
            for sample in full_train_ds:
                if isinstance(sample, dict):  # newer FLamby format
                    age_val = sample["features"][0].item()
                else:  # older FLamby format: (x, (t, e))
                    x, _ = sample
                    age_val = x[0].item()
                train_ages.append(age_val)

            age_mean = np.mean(train_ages)
            age_std = np.std(train_ages)

            # 2- Split the training dataset into 80/20
            full_train_list = list(full_train_ds)
            np.random.shuffle(full_train_list)
            split_idx = int(len(full_train_list) * 0.8)
            train_ds_split = full_train_list[:split_idx]
            val_ds_split = full_train_list[split_idx:]

            # 3- Ensure validation set has at least one event (E == 1)
            def get_event(sample):
                if isinstance(sample, dict):
                    return sample["event"]
                else:
                    _, y = sample
                    return int(y[1].item()) if hasattr(y[1], "item") else int(y[1])

            has_event_in_val = any(get_event(s) == 1 for s in val_ds_split)
            if not has_event_in_val:
                event_indices = [i for i, s in enumerate(train_ds_split) if get_event(s) == 1]
                if event_indices:
                    moved_sample = train_ds_split.pop(event_indices[0])
                    val_ds_split.append(moved_sample)
                    logger.warning(f"[Center {center}] Added 1 event sample from train → val to ensure event presence.")
                else:
                    logger.warning(f"[Center {center}] No events available in train to move to validation!")

            # Ensure there is at least one event (E == 1) in train, val, and test
            def ensure_event_presence(dataset, dataset_name):
                has_event = any(get_event(s) == 1 for s in dataset)
                if not has_event:
                    logger.warning(f"[{dataset_name}] No events found in the dataset!")

            ensure_event_presence(train_ds_split, f"Train (Center {center})")
            ensure_event_presence(val_ds_split, f"Validation (Center {center})")
            ensure_event_presence(test_ds, f"Test (Center {center})")

            # 4- Process datasets: normalize and extract numpy arrays
            def _process_dataset(dataset):
                features, time, event = [], [], []
                for sample in dataset:
                    if isinstance(sample, dict):  # new FLamby
                        x = sample["features"].clone().detach().numpy()
                        t = sample["time"]
                        e = sample["event"]
                    else:  # old FLamby
                        x, y = sample
                        x = x.clone().detach().numpy()
                        y_np = y.clone().detach().numpy()
                        t, e = y_np[0], y_np[1]

                    # Normalize age feature
                    if age_std > 0:
                        x[0] = (x[0] - age_mean) / age_std
                    else:
                        x[0] = 0.0

                    features.append(x[:39])
                    time.append(float(t))
                    event.append(int(e))

                return features, time, event

            # 5- Apply processing to splits
            train_features, train_time, train_event = _process_dataset(train_ds_split)
            val_features, val_time, val_event = _process_dataset(val_ds_split)
            test_features, test_time, test_event = _process_dataset(test_ds)

            msg = (
                f"[Center {center}] Split summary → "
                f"Train: {len(train_features)} samples ({sum(train_event)} events), "
                f"Val: {len(val_features)} samples ({sum(val_event)} events), "
                f"Test: {len(test_features)} samples ({sum(test_event)} events)"
            )
            log_and_print(msg)


            dataloaders[center] = {
                "train": {"features": train_features, "time": train_time, "event": train_event},
                "val": {"features": val_features, "time": val_time, "event": val_event},
                "test": {"features": test_features, "time": test_time, "event": test_event},
            }

        return dataloaders

    def get_local_dataloaders(self):
        """Create dataloaders for local training (single center)."""
        center = self.config.centers[0]
        train_ds = FedTcgaBrca(center=center, train=True)
        test_ds = FedTcgaBrca(center=center, train=False)
        return {
            "train": DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True),
            "test": DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False),
        }

    def get_centralized_dataloaders(self):
        """Create dataloaders for centralized training (all centers combined)."""
        train_datasets = [FedTcgaBrca(center=c, train=True) for c in self.config.centers]
        test_datasets = [FedTcgaBrca(center=c, train=False) for c in self.config.centers]

        train_ds = ConcatDataset(train_datasets)
        test_ds = ConcatDataset(test_datasets)

        return {
            "train": DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True),
            "test": DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False),
        }

    def get_dataloaders(self):
        """Return the appropriate dataloaders based on the training mode."""
        if self.config.training_mode == "federated":
            return self.get_federated_dataloaders()
        elif self.config.training_mode == "local":
            return self.get_local_dataloaders()
        elif self.config.training_mode == "centralized":
            return self.get_centralized_dataloaders()
        else:
            raise ValueError(f"Unsupported training mode: {self.config.training_mode}")
