# Training_Modes/Federated_Learning/clientRSF.py
import flwr as fl
import pickle
from flwr.common import FitRes, EvaluateRes, Parameters, Status, Code
from utils import evaluate_rsf   # you already have this
import numpy as np
from scipy.interpolate import interp1d

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
        #self.eval_times = np.array(config.eval_times_per_client[cid])
        # NEW --> GLOBAL OR CLIENT SPECIFIC EVAL TIME GRID
        if config.eval_grid_mode == "client":
            self.eval_times = np.array(config.eval_times_per_client[cid])

        elif config.eval_grid_mode == "global":
            self.eval_times = np.array(config.global_eval_times)

        else:
            raise ValueError("Unknown eval_grid_mode: must be 'client' or 'global'")    

        #DEBUG EVAL_TIMES:
        print(f"[DEBUG][Client {self.cid}] eval_grid_mode = {config.eval_grid_mode}")
        print(f"[DEBUG][Client {self.cid}] Raw eval_times BEFORE np.array constructor:")

        if config.eval_grid_mode == 'client':
            print(type(config.eval_times_per_client[self.cid]), config.eval_times_per_client[self.cid])
        elif config.eval_grid_mode == 'global':
            print(type(config.global_eval_times), config.global_eval_times)

        print(f"[DEBUG][Client {self.cid}] Final self.eval_times type={type(self.eval_times)}, len attempt...")

        try:
            print(f"[DEBUG][Client {self.cid}] len(self.eval_times) = {len(self.eval_times)}")
        except Exception as e:
            print(f"[ERROR][Client {self.cid}] self.eval_times HAS NO LENGTH! Type={type(self.eval_times)}, value={self.eval_times}")
            raise e


    # ---------------------------------------------------------
    # FIT: train local RSF and send trees to server
    # ---------------------------------------------------------
    def fit(self, ins):
        print(f"[DEBUG][Client {self.cid}] Starting FIT")

        # train local RSF:
        self.model.fit(self.X_train, self.y_train)
        trees = self.model.estimators_

        print(f"[DEBUG][Client {self.cid}] RSF trained, sending {len(self.model.estimators_)} trees")
    
        return FitRes(
            status=Status(Code.OK, message="OK"),
            parameters=Parameters(
                tensors=[pickle.dumps(trees)],
                tensor_type="pickle"
            ),
            num_examples=len(self.X_train),
            metrics={}
        )


    # EVALUATE: load global forest and compute metrics:

    def evaluate(self, ins):
        print(f"[DEBUG][Client {self.cid}] Starting EVALUATE")

        #DEBUG EVAL TIMES:
        print(f"[DEBUG][Client {self.cid}] eval_times at EVALUATE entry:")
        print(f"    type={type(self.eval_times)}, value={self.eval_times}")

        try:
            print(f"    len={len(self.eval_times)}")
        except Exception as e:
            print(f"[ERROR][Client {self.cid}] eval_times is unsized HERE")
            raise e
        #END DEBUG EVAL TIMES

        
        # server sends: [global_trees]
        federated_trees = pickle.loads(ins.parameters.tensors[0])
        print(f"[DEBUG][Client {self.cid}] Loaded global forest with {len(federated_trees)} trees")

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
        metrics["cid"] = self.cid

        print(f"[DEBUG][Client {self.cid}] Evaluation finished → {metrics}")

        return EvaluateRes(
            status=Status(Code.OK, message="OK"),
            loss=0.0,
            num_examples=len(self.X_test),
            metrics=metrics
        )