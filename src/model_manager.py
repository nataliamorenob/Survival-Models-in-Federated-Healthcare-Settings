from Models.CustomCoxModel import CustomCoxModel
from Models.CoxPH import CoxPH_model
from Models.StackedLogisticRegression import StackedLogisticRegression
from Models.FedSurF import SurvivalRandomForest
from Models.DeepSurv import DeepSurv

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
			
            # Both centralized and local use different number of trees (since we only have global forest and not local forests)
            n_estimators = (
                self.config.n_trees_federated
                if self.config.training_mode in ("centralized", "local")
                else self.config.n_trees_local
            )


            self.model = SurvivalRandomForest(
                n_estimators=n_estimators,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=client_seed)
        
        elif self.config.model == "DeepSurv":
            client_seed = self.config.random_state + self.client_id
            print(
                f"[DEBUG] Client {self.client_id} "
                f"global_seed={self.config.random_state} "
                f"client_seed={client_seed}"
            )
            
            # Initialize DeepSurv with config parameters
            self.model = DeepSurv(
                n_features=39,  # TCGA-BRCA has 39 features
                hidden_layers=getattr(self.config, 'deepsurv_hidden_layers', [64, 32, 16]),
                dropout=getattr(self.config, 'deepsurv_dropout', 0.3),
                batch_norm=getattr(self.config, 'deepsurv_batch_norm', True),
                activation=getattr(self.config, 'deepsurv_activation', 'ReLU'),
                lr=getattr(self.config, 'lr', 0.001),
                l2_reg=getattr(self.config, 'deepsurv_l2_reg', 0.0),
                epochs=getattr(self.config, 'num_epochs', 100),
                batch_size=getattr(self.config, 'batch_size', 64),
                random_state=client_seed
            )

        else: # TO DO for other models
            raise ValueError(f"Unsupported model type: {self.config.model}")

    def get_model(self):
        if self.model is None:
            raise RuntimeError("Model has not been initialized. Call initialize_model() first.")
        return self.model