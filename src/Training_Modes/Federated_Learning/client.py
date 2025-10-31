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
import numpy as np
import pandas as pd

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
        """Return model parameters."""
        import numpy as np
        import logging
        logger = logging.getLogger(__name__)

        if hasattr(self.model, "coef_") and hasattr(self.model, "intercept_"):
            return [self.model.coef_.astype(np.float64), self.model.intercept_.astype(np.float64)]

        # If model not yet fitted, initialize zeros with correct shape
        n_features = getattr(self, "n_features_in_", None)
        if n_features is None:
            n_features = 39  # fallback if not known
        return [np.zeros((1, n_features), dtype=np.float64), np.zeros(1, dtype=np.float64)]

    def set_parameters(self, parameters):
        """Receive parameters from server and store for initialization."""
        import numpy as np
        import logging
        logger = logging.getLogger(__name__)

        self.model.coef_ = np.array(parameters[0], dtype=np.float64)
        self.model.intercept_ = np.array(parameters[1], dtype=np.float64)
        self.model.classes_ = np.array([0, 1])  # Needed by sklearn for predict_proba
        self.fitted = True


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
                params = self.model.params_.reindex(ALL_FEATURE_COLUMNS).fillna(0).values
                logger.info(f"[Client {self.cid}] fit(): trained β mean={np.mean(params):.6f}, std={np.std(params):.6f}")
                return [params], len(self.train_data), {}
            
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

                # --- Return new parameters to Flower ---
                params = self.model.get_params()
                return params, len(X_train), {}



            else:
                raise NotImplementedError(f"Fit not implemented for model {self.config.model}")

        except Exception as e:
            logger.error(f"[Client {self.cid}] fit(): Exception {e}")
            logger.error(traceback.format_exc())
            # Return empty parameters and 0 samples to indicate failure
            if self.config.model.lower() == "coxph":
                return [np.zeros(len(ALL_FEATURE_COLUMNS))], 0, {}
            else:
                # For SLR, we need to know the shape of the parameters
                # This is a simplification. A more robust solution would be needed.
                return [np.zeros((1, self.train_data.shape[1] - 2))], 0, {}


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

