from utils import train, test
from flower.client import NumPyClient

class FLClient(NumPyClient):
    def __init__(self, model, train_data, test_data, config):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.config = config

    def get_parameters(self):
        # Return model parameters
        return self.model.get_params()

    def set_parameters(self, parameters):
        # Set model parameters
        self.model.set_params(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model = train(self.model, self.train_data, self.config)
        return self.get_parameters(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = test(self.model, self.test_data, self.config)
        return metrics["accuracy"], len(self.test_data), {}