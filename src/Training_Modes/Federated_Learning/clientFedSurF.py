# Training_Modes/Federated_Learning/clientRSF.py
import flwr as fl
import pickle
from flwr.common import FitRes, EvaluateRes, Parameters, Status, Code
from utils import evaluate_rsf   # you already have this
import numpy as np
from scipy.interpolate import interp1d
import os
from datetime import datetime
from sksurv.metrics import concordance_index_censored
#from Exps_runs_randomness.utils_results import append_metrics_to_csv #abajo descomentar tbn


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
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        self.X_test  = data["X_test"]
        self.y_test  = data["y_test"]

        # client-specific evaluation times: (old)
        #self.eval_times = np.array(config.eval_times_per_client[cid])



        # # NEW --> GLOBAL OR CLIENT SPECIFIC EVAL TIME GRID
        # if config.eval_grid_mode == "client":
        #     self.eval_times = np.array(config.eval_times_per_client[cid])

        # elif config.eval_grid_mode == "global":
        #     self.eval_times = np.array(config.global_eval_times)

        # else:
        #     raise ValueError("Unknown eval_grid_mode: must be 'client' or 'global'")    

        # #DEBUG EVAL_TIMES:
        # print(f"[DEBUG][Client {self.cid}] eval_grid_mode = {config.eval_grid_mode}")
        # print(f"[DEBUG][Client {self.cid}] Raw eval_times BEFORE np.array constructor:")

        # if config.eval_grid_mode == 'client':
        #     print(type(config.eval_times_per_client[self.cid]), config.eval_times_per_client[self.cid])
        # elif config.eval_grid_mode == 'global':
        #     print(type(config.global_eval_times), config.global_eval_times)

        # print(f"[DEBUG][Client {self.cid}] Final self.eval_times type={type(self.eval_times)}, len attempt...")

        # try:
        #     print(f"[DEBUG][Client {self.cid}] len(self.eval_times) = {len(self.eval_times)}")
        # except Exception as e:
        #     print(f"[ERROR][Client {self.cid}] self.eval_times HAS NO LENGTH! Type={type(self.eval_times)}, value={self.eval_times}")
        #     raise e




    # FIT: train local RSF and send trees to server
    # def fit(self, ins):
    #        '''
    #        This method is the old one (withoput the val set calculation for tree sampling)
    #        '''
    #     print(f"[DEBUG][Client {self.cid}] Starting FIT")

    #     # train local RSF:
    #     self.model.fit(self.X_train, self.y_train)
    #     trees = self.model.estimators_

    #     print(f"[DEBUG][Client {self.cid}] RSF trained, sending {len(self.model.estimators_)} trees")
    
    #     return FitRes(
    #         status=Status(Code.OK, message="OK"),
    #         parameters=Parameters(
    #             tensors=[pickle.dumps(trees)],
    #             tensor_type="pickle"
    #         ),
    #         num_examples=len(self.X_train),
    #         metrics={}
    #     )
    
    
    def fit(self, ins):
        print(f"[DEBUG][Client {self.cid}] Starting FIT (FedSurF-C)")

        # --------------------------------------------------
        # 1. Train local RSF
        # --------------------------------------------------
        self.model.fit(self.X_train, self.y_train)
        all_trees = self.model.estimators_

        print(f"[DEBUG][Client {self.cid}] Trained {len(all_trees)} local trees")

        # --------------------------------------------------
        # 2. Score each tree on VALIDATION set (plain C-index)
        # --------------------------------------------------
        cindices = []

        for tree in all_trees:
            try:
                surv_fns = tree.predict_survival_function(self.X_val)

                # simple scalar risk per sample (consistent & stable)
                risks = np.array([
                    -np.log(fn.y[-1] + 1e-8) for fn in surv_fns
                ])

                c = concordance_index_censored(
                    self.y_val["event"],
                    self.y_val["time"],
                    risks
                )[0]

            except Exception:
                c = 0.5  # neutral fallback

            cindices.append(c)

        cindices = np.array(cindices)
        cindices = np.nan_to_num(cindices, nan=0.5)
        cindices[cindices < 0.5] = 0.5  # safety


        # 3. Probabilistic tree sampling (FedSurF-C)
        probs = cindices / cindices.sum()

        n_send = self.config.n_trees_federated
        if n_send is None or n_send > len(all_trees):
            n_send = len(all_trees)

        selected_idx = np.random.choice(
            len(all_trees),
            size=n_send,
            replace=False,
            p=probs
        )

        selected_trees = [all_trees[i] for i in selected_idx]

        print(
            f"[DEBUG][Client {self.cid}] "
            f"Selected {len(selected_trees)} trees "
            f"(mean val C-index={cindices[selected_idx].mean():.3f})"
        )

        # 4. Send ONLY selected trees to server
        return FitRes(
            status=Status(Code.OK, message="OK"),
            parameters=Parameters(
                tensors=[pickle.dumps(selected_trees)],
                tensor_type="pickle"
            ),
            num_examples=len(self.X_train),
            metrics={}
        )



    # EVALUATE: load global forest and compute metrics:

    def evaluate(self, ins):
        print(f"[DEBUG][Client {self.cid}] Starting EVALUATE")

        #DEBUG EVAL TIMES:
        #print(f"[DEBUG][Client {self.cid}] eval_times at EVALUATE entry:")
        #print(f"    type={type(self.eval_times)}, value={self.eval_times}")

        # try:
        #     #print(f"    len={len(self.eval_times)}")
        # except Exception as e:
        #     print(f"[ERROR][Client {self.cid}] eval_times is unsized HERE")
        #     raise e
        #END DEBUG EVAL TIMES

        
        # server sends: [global_trees]
        federated_trees = pickle.loads(ins.parameters.tensors[0])
        print(f"[DEBUG][Client {self.cid}] Loaded global forest with {len(federated_trees)} trees")

        # load global forest
        n_features = self.X_train.shape[1]
        #self.model.set_trees(federated_trees, n_features)
        self.model.set_trees(
            trees=federated_trees,
            n_features=n_features,
            init_times=self.config.union_time_grid
        )


        # evaluate on local grid
        metrics = evaluate_rsf(
            model=self.model,
            data={
                "X_test": self.X_test,
                "y_test": self.y_test,
                "y_train": self.y_train,
                #"eval_times": self.eval_times
            },
            client_id=self.cid,
            config=self.config,
        )
        metrics["cid"] = self.cid

        print(f"[DEBUG][Client {self.cid}] Evaluation finished → {metrics}")


        # TO CSV OF RANDOMNESS:
        # SAVE FINAL METRICS (ONE ROW PER CLIENT PER RUN) -->
        run_id = os.environ.get("RUN_ID", "unknown")

        csv_path = os.environ.get(
            "OUTPUT_CSV",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "results_randomness_exps",
                f"run_{run_id}.csv"
            )
        )


        # append_metrics_to_csv(
        #     csv_path,
        #     {
        #         "timestamp": datetime.now().isoformat(),
        #         "run_id": run_id,
        #         "client_id": self.cid,
        #         "c_index": metrics["C-index"],
        #         "auc": metrics["AUC"],
        #         "ibs": metrics["IBS"],
        #     }
        # )




        return EvaluateRes(
            status=Status(Code.OK, message="OK"),
            loss=0.0,
            num_examples=len(self.X_test),
            metrics=metrics
        )