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

from config import ALL_FEATURE_COLUMNS

class FederatedCoxClient(NumPyClient):
    def __init__(self, cid, name, model, dataset_manager, config):
        self.cid = cid
        self.name = name
        self.model_fn = model # model constructor/function (CoxPH_model)
        self.model = None # will store the trained model instance
        self.dataset_manager = dataset_manager
        self.config = config

        # Load data using dataset_manager:
        self.dataloaders = self.dataset_manager.get_federated_dataloaders()
        self.center = list(self.dataloaders.keys())[0]  
        self.train_data = self.dataloaders[self.center]["train"]
        self.val_data = self.dataloaders[self.center]["val"] # Returns None for CoxPH_model
        self.test_data = self.dataloaders[self.center]["test"]

        # For CoxPH model:
        # self.fitted_model = None # holds the trained lifelines.CoxPHFitter model after training
        # # For deep models: 
        # self.weights = None
        # self.local_beta = None

    def get_parameters(self, config=None):
        #"""Return model parameters (fixed effects coefficients)."""
        #return self.model.get_coefficients()
        """Return model parameters depending on model type."""
        if self.config.model.lower() == "coxph":
            if self.model is not None:
                # Return parameters aligned with the global feature list
                params = self.model.params_.reindex(ALL_FEATURE_COLUMNS).fillna(0)
                return [params.values]
            else: 
                # Return zeros for all feature columns
                return [np.zeros(len(ALL_FEATURE_COLUMNS))]
        else: # TO DO
            # Deep models should implement get_weights()
            return self.model.get_weights()

    def set_parameters(self, parameters):
        # """Set model parameters (fixed effects coefficients)."""
        # self.model.beta = parameters[0]
        if self.config.model.lower() == "coxph":
            # Lifelines CoxPHFitter does not allow direct coefficient injection
            if parameters:
                self.local_beta = parameters[0]
            else:
                self.local_beta = None
        else: # TO DO
            # Deep model parameters
            self.model.set_weights(parameters)

    def fit(self, parameters, config):
        """Train the model on local data using the generic train function."""
        self.set_parameters(parameters)
        
        if self.config.model.lower() == "coxph":
            # Train CoxPH on DataFrame
            fitted_model = self.model_fn(
                self.train_data,
                config=self.config,
                client_id=self.cid,
                duration_col="time",
                event_col="event",
                init_params=self.local_beta,
            )
            self.model = fitted_model
            
            logging.info(
                f"[Client {self.cid}] β mean={self.model.params_.mean():.4f}, "
                f"std={self.model.params_.std():.4f}, "
                f"min={self.model.params_.min():.4f}, "
                f"max={self.model.params_.max():.4f}"
            )
        else: # TO DO
            # Train deep survival model
            train_model(self.model, self.train_data, self.val_data, self.config)

        num_examples = len(self.train_data)
        return self.get_parameters(), num_examples, {}

    def evaluate(self, parameters, config):
        """Evaluate the model on local test data using the generic evaluate function."""
        self.set_parameters(parameters)

        if self.config.model.lower() == "coxph":
            # In evaluation, we want to use the global model parameters.
            # We create a temporary model and fit it with the global parameters as an initial point.
            eval_model = self.model_fn(
                self.train_data,
                config=self.config,
                client_id=self.cid,
                duration_col="time",
                event_col="event",
                init_params=self.local_beta,
            )

            metrics = evaluate_model(
                eval_model,
                self.test_data,
                self.config,
                train_data=self.train_data,
                client_id=self.cid,
            )
        else:
            if self.model is None:
                raise RuntimeError("Deep model is None at evaluation!")
            metrics = evaluate_model(self.model, self.test_data, self.config, train_data=self.train_data)

        num_examples = len(self.test_data)
        loss = 0.0
        return loss, num_examples, metrics

    def get_random_effect(self):
        """Return the local random effect."""
        return self.model.get_random_effect()

