from flwr.client import NumPyClient
from utils import evaluate_rsf
import logging


class FederatedRSFClient(NumPyClient):
    def __init__(self, cid, name, model, config, dataloaders):
        self.cid = cid
        self.name = name
        self.config = config
        self.model = model   
        
        # dataloaders = {center_id : {...}}
        center = list(dataloaders.keys())[0]
        center_data = dataloaders[center]

        # Extract preprocessed arrays
        self.X_train = center_data["X_train"]
        self.y_train = center_data["y_train"]
        self.X_test  = center_data["X_test"]
        self.y_test  = center_data["y_test"]
        self.eval_times = center_data["eval_times"]

        # Keep dfs for debugging (optional)
        self.train_df = center_data["train_df"]
        self.test_df  = center_data["test_df"]

        self.logger = logging.getLogger("main")

    # ---------------------------------------------------------
    # RSF does not send parameters
    # ---------------------------------------------------------
    def get_parameters(self, config=None):
        return []

    # ---------------------------------------------------------
    # LOCAL TRAINING
    # ---------------------------------------------------------
    def fit(self, parameters, config):
        self.logger.info(
            f"[Client {self.cid}] Training local RSF with {len(self.X_train)} samples."
        )

        # Fit model using scikit-survival RSF
        self.model.fit(self.X_train, self.y_train)

        local_trees = self.model.estimators_

        return local_trees, len(self.X_train), {}

    # ---------------------------------------------------------
    # LOCAL EVALUATION
    # ---------------------------------------------------------
    def evaluate(self, parameters, config):
        # Parameters contains {"trees": [...]}
        self.model.estimators_ = parameters["trees"]

        metrics = evaluate_rsf(
            model=self.model,
            data={
                "X_test": self.X_test,
                "y_test": self.y_test,
                "eval_times": self.eval_times,
            },
            client_id=self.cid,
            config=self.config,
        )

        return 0.0, len(self.X_test), metrics