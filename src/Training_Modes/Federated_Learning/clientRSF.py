from flwr.client import Client
from utils import evaluate_rsf
import logging
import flwr as fl
from flwr.common import (
    Parameters,
    GetParametersRes,
    FitRes,
    EvaluateRes,
    Status,
    Code,
)
from flwr.common import Parameters, GetParametersRes, Status, Code
import pickle
# class FederatedRSFClient(NumPyClient):
#     def __init__(self, cid, name, model, config, dataloaders):
#         self.cid = cid
#         self.name = name
#         self.config = config
#         self.model = model   
        
#         # dataloaders = {center_id : {...}}
#         center = list(dataloaders.keys())[0]
#         center_data = dataloaders[center]

#         # Extract preprocessed arrays
#         self.X_train = center_data["X_train"]
#         self.y_train = center_data["y_train"]
#         self.X_test  = center_data["X_test"]
#         self.y_test  = center_data["y_test"]
#         self.eval_times = center_data["eval_times"]

#         # Keep dfs for debugging (optional)
#         self.train_df = center_data["train_df"]
#         self.test_df  = center_data["test_df"]

#         self.logger = logging.getLogger("main")

#     # ---------------------------------------------------------
#     # RSF does not send parameters
#     # ---------------------------------------------------------
#     def get_parameters(self, config=None):
#         return []

#     # ---------------------------------------------------------
#     # LOCAL TRAINING
#     # ---------------------------------------------------------
#     def fit(self, parameters, config):
#         self.logger.info(
#             f"[Client {self.cid}] Training local RSF with {len(self.X_train)} samples."
#         )

#         # Fit model using scikit-survival RSF
#         self.model.fit(self.X_train, self.y_train)

#         # DEBUG: Cheeck how many trees have been build
#         n_trees = len(self.model.estimators_) # number of trees built
#         print(f"[Client {self.cid}] Built {n_trees} trees.")
#         local_trees = self.model.estimators_

#         return local_trees, len(self.X_train), {}

#     # ---------------------------------------------------------
#     # LOCAL EVALUATION
#     # ---------------------------------------------------------
#     def evaluate(self, parameters, config):
#         self.model.set_trees(parameters) # Parameters is a list of sklearn trees

#         metrics = evaluate_rsf(
#             model=self.model,
#             data={
#                 "X_test": self.X_test,
#                 "y_test": self.y_test,
#                 "y_train": self.y_train,
#                 "eval_times": self.eval_times,
#             },
#             client_id=self.cid,
#             config=self.config,
#         )

#         return 0.0, len(self.X_test), metrics



class FederatedRSFClient(Client):
    def __init__(self, cid, name, model, config, dataloaders):
        self.cid = cid
        self.name = name
        self.model = model
        self.config = config

        center = list(dataloaders.keys())[0]
        data = dataloaders[center]

        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_test = data["X_test"]
        self.y_test = data["y_test"]
        self.eval_times = data["eval_times"]

    # GET PARAMETERS (not used):
    from flwr.common import Parameters, GetParametersRes


    def get_parameters(self, ins):
        # For RSF, we use get_parameters to return only event times in round 0
        if ins.config.get("request_event_times", False):
            event_times = self.get_event_times()
            return fl.common.GetParametersRes(
                status=fl.common.Status(code=fl.common.Code.OK),
                parameters=fl.common.ndarrays_to_parameters([np.array(event_times)])
            )

        # Otherwise, return empty parameters
        empty = Parameters(tensors=[], tensor_type="")
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=empty)



    def fit(self, ins):

        # Case 1: Round 0 → send event times
        if ins.config.get("send_event_times", False):
            event_times = self.y_train["time"][self.y_train["event"]].astype(float)
            return fl.common.FitRes(
                status=fl.common.Status(code=fl.common.Code.OK),
                parameters=fl.common.ndarrays_to_parameters([event_times]),
                num_examples=len(event_times),
                metrics={}
            )

        # Case 2: Normal RSF training
        self.model.fit(self.X_train, self.y_train)
        trees = self.model.estimators_

        return FitRes(
            status=Status(code=Code.OK),
            parameters=Parameters(
                tensors=[pickle.dumps(trees)],
                tensor_type="pickle",
            ),
            num_examples=len(self.X_train),
            metrics={}
        )



    def evaluate(self, ins):
        #trees = ins.parameters # list of sklearn trees
        #self.model.set_trees(trees)
        
        import pickle
        # trees = pickle.loads(ins.parameters.tensors[0])
        # n_features = self.X_train.shape[1]
        # self.model.set_trees(trees, n_features)
        federated_trees = pickle.loads(ins.parameters.tensors[0])
        global_event_times = pickle.loads(ins.parameters.tensors[1])

        n_features = self.X_train.shape[1]
        self.model.set_trees(
            trees=federated_trees,
            n_features=n_features,
            global_event_times=global_event_times)


        metrics = evaluate_rsf(
            model=self.model,
            data={
                "X_test": self.X_test,
                "y_test": self.y_test,
                "y_train": self.y_train,
                "eval_times": self.eval_times
            },
            client_id=self.cid,
            config=self.config,
        )

        return fl.common.EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK),
            loss=0.0,
            num_examples=len(self.X_test),
            metrics=metrics
        )


    def get_event_times(self):
        """Return client training event times for global synchronization."""
        times = self.y_train["time"][self.y_train["event"]].astype(float).tolist()
        return times
