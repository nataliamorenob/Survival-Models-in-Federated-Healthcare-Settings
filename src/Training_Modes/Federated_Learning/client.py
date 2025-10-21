from utils import train, test
import flwr as fl
from fl.client import NumPyClient
from dataset_manager import DatasetManager
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
from Training_Modes.Federated_Learning.task import get_weights, set_weights
import logging
from utils import train_model, evaluate_model

class FLClient(NumPyClient):
    def __init__(self, model, dataset_manager, config):
        self.model = model
        self.config = config


        self.dataset_manager = dataset_manager
        self.train_loader = self.dataset_manager.get_dataloader("train", batch_size=config.batch_size, shuffle=True)
        self.val_loader = self.dataset_manager.get_dataloader("val", batch_size=config.val_batch_size, shuffle=False)
        self.test_loader = self.dataset_manager.get_dataloader("test", batch_size=config.test_batch_size, shuffle=False)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_weights(self.model)

    # def set_parameters(self, parameters):
    #     # Set model parameters
    #     self.model.set_params(parameters)

    def fit(self, parameters, config):
        logger = logging.getLogger(__name__)
        logger.info("Starting federated training...")

        #self.set_parameters(parameters)
        self.set_weights(self.model, parameters)

        metrics = train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config,
        )
        logger.info(f"Finished federated training...")

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        #self.set_parameters(parameters)
        logger = logging.getLogger(__name__)
        logger.info("Starting federated evaluation...")

        self.set_weights(self.model, parameters)
        metrics = evaluate_model(self.model, self.test_loader, self.config)
        return metrics["accuracy"], len(self.test_loader.dataset), {}
    
