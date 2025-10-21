from flamby.datasets.fed_tcga_brca import FedTcgaBrca
from torch.utils.data import DataLoader, ConcatDataset

class DatasetManager:
    def __init__(self, config):
        """
        Initialize the DatasetManager with the given configuration.
        """
        self.config = config

    def get_federated_dataloaders(self):
        """
        Create dataloaders for federated training.
        """
        dataloaders = {}
        for center in self.config.centers:
            train_ds = FedTcgaBrca(center=center, train=True)
            test_ds = FedTcgaBrca(center=center, train=False)
            dataloaders[center] = {
                "train": DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True),
                "test": DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False)
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