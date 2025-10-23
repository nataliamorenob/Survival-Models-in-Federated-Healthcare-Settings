import flwr as fl
from flwr.client import NumPyClient
from dataset_manager import DatasetManager
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
from Training_Modes.Federated_Learning.task import get_weights, set_weights
import logging
from utils import train_model, evaluate_model
from Models.CustomCoxModel import CustomCoxModel
import numpy as np

# class FLClient(NumPyClient):
#     def __init__(self, model, dataset_manager, config):
#         self.model = model
#         self.config = config


#         self.dataset_manager = dataset_manager
#         self.train_loader = self.dataset_manager.get_dataloader("train", batch_size=config.batch_size, shuffle=True)
#         self.val_loader = self.dataset_manager.get_dataloader("val", batch_size=config.val_batch_size, shuffle=False)
#         self.test_loader = self.dataset_manager.get_dataloader("test", batch_size=config.test_batch_size, shuffle=False)

#     def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
#         return get_weights(self.model)

#     # def set_parameters(self, parameters):
#     #     # Set model parameters
#     #     self.model.set_params(parameters)

#     def fit(self, parameters, config):
#         logger = logging.getLogger(__name__)
#         logger.info("Starting federated training...")

#         #self.set_parameters(parameters)
#         self.set_weights(self.model, parameters)

#         metrics = train_model(
#             model=self.model,
#             train_loader=self.train_loader,
#             val_loader=self.val_loader,
#             config=self.config,
#         )
#         logger.info(f"Finished federated training...")

#         return self.get_parameters(), len(self.train_loader.dataset), {}

#     def evaluate(self, parameters, config):
#         #self.set_parameters(parameters)
#         logger = logging.getLogger(__name__)
#         logger.info("Starting federated evaluation...")

#         self.set_weights(self.model, parameters)
#         metrics = evaluate_model(self.model, self.test_loader, self.config)
#         return metrics["accuracy"], len(self.test_loader.dataset), {}

class FederatedCoxClient(NumPyClient):
    def __init__(self, dataset_manager, config):
        self.model = CustomCoxModel()
        self.config = config
        self.dataset_manager = dataset_manager

        # Load data using dataset_manager:
        self.dataloaders = self.dataset_manager.get_federated_dataloaders()
        self.center = list(self.dataloaders.keys())[0]  
        self.train_data = self.dataloaders[self.center]["train"]
        self.val_data = self.dataloaders[self.center]["val"]
        self.test_data = self.dataloaders[self.center]["test"]


    def get_parameters(self, config=None):
        """Return model parameters (fixed effects coefficients)."""
        return self.model.get_coefficients()

    def set_parameters(self, parameters):
        """Set model parameters (fixed effects coefficients)."""
        self.model.beta = np.array(parameters)

    def fit(self, parameters, config):
        """Train the model on local data using the generic train function."""
        self.set_parameters(parameters)
        
        # Use the dispatcher function from utils.py
        train_model(self.model, self.train_data, self.val_data, self.config)

        num_examples = len(self.train_data["features"])
        return self.get_parameters(), num_examples, {}

    def evaluate(self, parameters, config):
        """Evaluate the model on local test data using the generic evaluate function."""
        self.set_parameters(parameters)

        # Use the dispatcher function from utils.py
        metrics = evaluate_model(self.model, self.test_data, self.config)
        
        num_examples = len(self.test_data["features"])
        
        # Flower's evaluate function expects a loss value to be returned.
        # We can return a placeholder and pass the real metrics in the dictionary.
        loss = 0.0 
        
        return loss, num_examples, metrics

    def get_random_effect(self):
        """Return the local random effect."""
        return self.model.get_random_effect()

