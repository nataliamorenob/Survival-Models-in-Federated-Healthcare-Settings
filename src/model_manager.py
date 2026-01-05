from Models.CustomCoxModel import CustomCoxModel
from Models.CoxPH import CoxPH_model
from Models.StackedLogisticRegression import StackedLogisticRegression
from Models.FedSurF import SurvivalRandomForest

class ModelManager:
    def __init__(self, config, client_id):
        self.config = config
        self.model = None
        self.client_id = client_id

    def initialize_model(self):
        if self.config.model == "CoxPH":
            self.model = CoxPH_model
        elif self.config.model == "SLR":
            self.model = StackedLogisticRegression()
        elif self.config.model == "RSF":
            client_seed = self.config.random_state + self.client_id
            print(
                f"[DEBUG] Client {self.client_id} "
                f"global_seed={self.config.random_state} "
                f"client_seed={client_seed}"
            )
            
            self.model = SurvivalRandomForest(
                n_estimators=self.config.n_trees_local,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=client_seed)

        else: # TO DO for other models
            raise ValueError(f"Unsupported model type: {self.config.model}")

    def get_model(self):
        if self.model is None:
            raise RuntimeError("Model has not been initialized. Call initialize_model() first.")
        return self.model