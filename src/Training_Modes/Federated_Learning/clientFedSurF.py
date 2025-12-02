# Training_Modes/Federated_Learning/clientRSF.py
import flwr as fl
import pickle
from flwr.common import FitRes, EvaluateRes, Parameters, Status, Code
from utils import evaluate_rsf   # you already have this
import numpy as np

class FederatedRSFClient(fl.client.Client):

    def __init__(self, cid, name, model, config, dataloaders):
        self.cid = cid
        self.name = name
        self.model = model
        self.config = config

        center = list(dataloaders.keys())[0]
        data = dataloaders[center]

        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_test  = data["X_test"]
        self.y_test  = data["y_test"]

        # client-specific evaluation times
        self.eval_times = np.array(config.eval_times_per_client[cid])

    # ---------------------------------------------------------
    # FIT: train local RSF and send trees to server
    # ---------------------------------------------------------
    def fit(self, ins):

        # train local RSF
        self.model.fit(self.X_train, self.y_train)
        trees = self.model.estimators_

        return FitRes(
            status=Status(Code.OK, message="OK"),
            parameters=Parameters(
                tensors=[pickle.dumps(trees)],
                tensor_type="pickle"
            ),
            num_examples=len(self.X_train),
            metrics={}
        )

    # ---------------------------------------------------------
    # EVALUATE: load global forest and compute metrics
    # ---------------------------------------------------------
    def evaluate(self, ins):

        # server sends: [global_trees]
        federated_trees = pickle.loads(ins.parameters.tensors[0])

        # load global forest
        n_features = self.X_train.shape[1]
        self.model.set_trees(federated_trees, n_features)

        # evaluate on local grid
        metrics = evaluate_rsf(
            model=self.model,
            data={
                "X_test": self.X_test,
                "y_test": self.y_test,
                "y_train": self.y_train,
                "eval_times": self.eval_times},
            client_id=self.cid,
            config=self.config,
        )

        return EvaluateRes(
            status=Status(Code.OK, message="OK"),
            loss=0.0,
            num_examples=len(self.X_test),
            metrics=metrics
        )