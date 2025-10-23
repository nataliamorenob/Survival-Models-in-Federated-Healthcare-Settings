from flamby.datasets.fed_tcga_brca import FedTcgaBrca
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

class DatasetManager:
    def __init__(self, config):
        """
        Initialize the DatasetManager with the given configuration.
        """
        self.config = config

    def get_federated_dataloaders(self):
        """
        Create dataloaders for federated training. This method now splits the
        original training data into an 80% training set and a 20% validation set.
        It also scales the age feature based on the mean/std of the original training set.
        """
        dataloaders = {}
        for center in self.config.centers:
            # Load the full training and test sets for the center
            full_train_ds = FedTcgaBrca(center=center, train=True)
            test_ds = FedTcgaBrca(center=center, train=False)

            # 1. Calculate scaling parameters from the entire original training set
            train_ages = [sample["features"][0] for sample in full_train_ds]
            age_mean = np.mean(train_ages)
            age_std = np.std(train_ages)

            # 2. Split the original training data into training and validation sets
            # Convert to list to shuffle and split
            full_train_list = list(full_train_ds)
            np.random.shuffle(full_train_list)  # Shuffle for a random split
            
            split_idx = int(len(full_train_list) * 0.8)
            train_ds_split = full_train_list[:split_idx]
            val_ds_split = full_train_list[split_idx:]

            # 3. Helper function to process datasets (apply scaling)
            def _process_dataset(dataset):
                features = []
                for sample in dataset:
                    processed_features = sample["features"].copy()
                    # Scale age feature (feature_0) using params from the full training set
                    if age_std > 0:
                        processed_features[0] = (processed_features[0] - age_mean) / age_std
                    else:
                        processed_features[0] = 0  # Handle case where std is zero
                    features.append(processed_features[:39])
                
                time = [sample["time"] for sample in dataset]
                event = [sample["event"] for sample in dataset]
                return features, time, event

            # 4. Process the new train, validation, and test sets
            train_features, train_time, train_event = _process_dataset(train_ds_split)
            val_features, val_time, val_event = _process_dataset(val_ds_split)
            test_features, test_time, test_event = _process_dataset(test_ds)

            dataloaders[center] = {
                "train": {
                    "features": train_features,
                    "time": train_time,
                    "event": train_event
                },
                "val": {
                    "features": val_features,
                    "time": val_time,
                    "event": val_event
                },
                "test": {
                    "features": test_features,
                    "time": test_time,
                    "event": test_event
                }
            }
        return dataloaders

    def get_local_dataloaders(self):
        """
        Create dataloaders for local training (single center).
        """
        center = self.config.centers[0]  # Assuming the first center for local training
        train_ds = FedTcgaBrca(center=center, train=True)
        test_ds = FedTcgaBrca(center=center, train=False)
        return {
            "train": DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True),
            "test": DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False)
        }

    def get_centralized_dataloaders(self):
        """
        Create dataloaders for centralized training (all centers combined).
        """
        train_datasets = [FedTcgaBrca(center=center, train=True) for center in self.config.centers]
        test_datasets = [FedTcgaBrca(center=center, train=False) for center in self.config.centers]

        train_ds = ConcatDataset(train_datasets)
        test_ds = ConcatDataset(test_datasets)

        return {
            "train": DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True),
            "test": DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False)
        }

    def get_dataloaders(self):
        """
        Return the appropriate dataloaders based on the training mode.
        """
        if self.config.training_mode == "federated":
            return self.get_federated_dataloaders()
        elif self.config.training_mode == "local":
            return self.get_local_dataloaders()
        elif self.config.training_mode == "centralized":
            return self.get_centralized_dataloaders()
        else:
            raise ValueError(f"Unsupported training mode: {self.config.training_mode}")