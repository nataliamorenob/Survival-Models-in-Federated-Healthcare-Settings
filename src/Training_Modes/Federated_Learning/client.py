import flwr as fl
from flwr.client import NumPyClient
from dataset_manager import DatasetManager
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
from Training_Modes.Federated_Learning.task import get_weights, set_weights
import logging
from utils import train_model, evaluate_model
from Models.CustomCoxModel import CustomCoxModel
from Models.StackedLogisticRegression import StackedLogisticRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

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
    def __init__(self, cid, name, model, dataset_manager, config, dataloaders):
        self.cid = cid
        self.name = name
        self.model_fn = model # model constructor/function (CoxPH_model)
        self.model = model if config.model.lower() != "coxph" else None # will store the trained model instance
        self.dataset_manager = dataset_manager
        self.config = config

        # Load data using dataset_manager:
        #self.dataloaders = self.dataset_manager.get_federated_dataloaders()
        self.dataloaders = dataloaders
        self.center = list(self.dataloaders.keys())[0]  
        self.train_data = self.dataloaders[self.center]["train"]
        self.val_data = self.dataloaders[self.center]["val"]
        self.test_data = self.dataloaders[self.center]["test"]

        # For CoxPH model:
        self.local_beta = None
        self.logger = logging.getLogger("main")


    def get_parameters(self, config=None):
        """Return model parameters as a Flower Parameters object."""
        # Ensure model exists and is fitted
        if not hasattr(self, "model") or not getattr(self.model, "fitted", False):
            n_features = self.train_data.shape[1] - 1  # exclude event column
            self.logger.info(
                f"[Client {self.cid}] get_parameters: model not fitted, returning zero params for {n_features} features."
            )
            zeros = [np.zeros((1, n_features), dtype=np.float32), np.zeros(1, dtype=np.float32)]
            return zeros

        # Extract coefficients/intercept from sklearn model
        coef = self.model.model.coef_.astype(np.float32)
        intercept = self.model.model.intercept_.astype(np.float32)

        self.logger.info(
            f"[Client {self.cid}] get_parameters: mean={coef.mean():.6f}, std={coef.std():.6f}"
        )

        return [coef, intercept]

    def set_parameters(self, parameters):
        """Receive global parameters from the server and load into local model."""
        from flwr.common import Parameters

        # Convert Flower Parameters to NumPy arrays if needed
        if isinstance(parameters, Parameters):
            ndarrays = parameters_to_ndarrays(parameters)
        else:
            ndarrays = parameters

        coef, intercept = ndarrays
        if not hasattr(self.model, "model"):
            self.model.model = LogisticRegression(warm_start=True, solver="liblinear")

        # Load parameters into model
        self.model.model.coef_ = coef.reshape(1, -1)
        self.model.model.intercept_ = intercept.reshape(1,)
        self.model.model.classes_ = np.array([0, 1])
        self.model.fitted = True

        self.logger.info(
            f"[Client {self.cid}] set_parameters: loaded weights mean={coef.mean():.6f}, std={coef.std():.6f}"
        )

    def fit(self, parameters, config):
        import traceback
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[Client {self.cid}] fit(): starting local training")

        try:
            self.set_parameters(parameters)

            if self.config.model.lower() == "coxph":
                if self.local_beta is not None:
                    logger.info(f"[Client {self.cid}] fit(): received β mean={np.mean(self.local_beta):.6f}")
                
                fitted_model = self.model_fn(
                    self.train_data,
                    config=self.config,
                    client_id=self.cid,
                    duration_col="time",
                    event_col="event",
                    init_params=self.local_beta,
                )
                self.model = fitted_model
                params = [
                    self.model.params_.reindex(ALL_FEATURE_COLUMNS).fillna(0).values.astype(np.float32)
                ]

                logger.info(
                    f"[Client {self.cid}] fit(): trained β mean={np.mean(params[0]):.6f}, std={np.std(params[0]):.6f}"
                )
                return params, len(self.train_data), {}
            
            elif self.config.model.lower() == "slr":
                X_train = self.train_data.drop(columns=["event"])
                y_train = self.train_data["event"]

                # --- Defensive logging before training ---
                if hasattr(self.model.model, "coef_"):
                    self.logger.info(
                        f"[Client {self.cid}] Fit called — existing weights mean={self.model.model.coef_.mean():.6f}, "
                        f"std={self.model.model.coef_.std():.6f}"
                    )
                else:
                    self.logger.info(f"[Client {self.cid}] Fit called — model uninitialized (first round).")

                # --- Local training ---
                self.model.fit(X_train, y_train)

                # --- Log new weights after training ---
                self.logger.info(
                    f"[Client {self.cid}] Model coef (after fit): mean={self.model.model.coef_.mean():.6f}, "
                    f"std={self.model.model.coef_.std():.6f}"
                )

                params = [
                    self.model.model.coef_.astype(np.float32),
                    self.model.model.intercept_.astype(np.float32),
                ]
                return params, len(X_train), {}

            else:
                raise NotImplementedError(f"Fit not implemented for model {self.config.model}")

        except Exception as e:
            logger.error(f"[Client {self.cid}] fit(): Exception {e}")
            logger.error(traceback.format_exc())

            n_features = self.train_data.shape[1] - 2
            empty_params = [
                np.zeros((1, n_features), dtype=np.float32),
                np.zeros(1, dtype=np.float32),
            ]
            return empty_params, 0, {}


    def evaluate(self, parameters, config):
        """Evaluate the model on local test data."""
        self.set_parameters(parameters)

        if self.config.model.lower() == "coxph":
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
        
        elif self.config.model.lower() == "slr":
            X_test = self.test_data.drop(columns=["event"])
            y_test = self.test_data["event"]

            hazards = self.model.predict_hazard(X_test)

            # Compute metrics externally (C-Index, IBS, AUC)
            metrics = evaluate_model(self.model, self.test_data, self.config, train_data=self.train_data)
            return 0.0, len(X_test), metrics

        else:
            raise NotImplementedError(f"Evaluate not implemented for model {self.config.model}")

        num_examples = len(self.test_data)
        loss = 0.0  # We don't have a standard loss for CoxPH/SLR in this context
        return loss, num_examples, metrics

    def get_random_effect(self):
        """Return the local random effect."""
        if self.config.model.lower() == "coxph":
            return self.model.get_random_effect()
        return None

