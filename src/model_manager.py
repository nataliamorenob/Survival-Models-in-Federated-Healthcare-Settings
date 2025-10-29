from Models.CustomCoxModel import CustomCoxModel
from Models.CoxPH import CoxPH_model
from Models.StackedLogisticRegression import StackedLogisticRegression

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.model = None

    def initialize_model(self):
        if self.config.model == "CoxPH":
            self.model = CoxPH_model
        elif self.config.model == "SLR":
            self.model = StackedLogisticRegression()
        else: # TO DO for other models
            raise ValueError(f"Unsupported model type: {self.config.model}")

    def get_model(self):
        if self.model is None:
            raise RuntimeError("Model has not been initialized. Call initialize_model() first.")
        return self.model