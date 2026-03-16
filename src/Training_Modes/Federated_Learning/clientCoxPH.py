# Training_Modes/Federated_Learning/clientCoxPH.py
"""
Federated CoxPH client using local risk set approximation.

This client is dedicated to the linear FLamby-style CoxPH model so its
federated flow lives separately from DeepSurv, even though both models share
the same survival-loss training pattern.
"""

import os
from datetime import datetime

import flwr as fl
import numpy as np
import torch
from flwr.common import Code, EvaluateRes, FitRes, GetParametersRes, NDArrays, Status

from Exps_runs_randomness.utils_results import append_metrics_to_csv
from utils import evaluate_deepsurv


class FederatedCoxPHClient(fl.client.Client):
    """
    Federated learning client for the linear neural CoxPH model.

    As in FLamby-style federated Cox training, the risk sets are local to the
    client data seen during optimization.
    """

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
        self.X_test = data["X_test"]
        self.y_test = data["y_test"]

    def get_parameters(self, ins=None):
        from flwr.common import ndarrays_to_parameters

        parameters_proto = ndarrays_to_parameters(self._get_parameters_as_arrays())
        return GetParametersRes(
            status=Status(Code.OK, message="OK"),
            parameters=parameters_proto,
        )

    def _get_parameters_as_arrays(self):
        return [param.cpu().detach().numpy() for param in self.model.network.state_dict().values()]

    def set_parameters(self, parameters: NDArrays) -> None:
        state_dict = self.model.network.state_dict()
        for param_tensor, new_value in zip(state_dict.values(), parameters):
            param_tensor.copy_(torch.tensor(new_value))

    def _reshape_flower_parameters(self, parameters):
        param_idx = 0
        state_dict = self.model.network.state_dict()
        params_list = []

        for _, param in state_dict.items():
            param_shape = param.shape
            param_size = param.numel()
            params_list.append(parameters[param_idx][:param_size].reshape(param_shape))
            param_idx += 1

        return params_list

    def fit(self, ins):
        dataset_size = len(self.X_train)
        batch_size = self.model.batch_size
        updates_per_epoch = dataset_size / float(batch_size)

        if hasattr(self.model, "num_updates_per_round") and self.model.num_updates_per_round:
            expected_epochs = int(np.ceil(self.model.num_updates_per_round / updates_per_epoch))
            print(
                f"[Client {self.cid}] Training CoxPH: "
                f"{self.model.num_updates_per_round} updates ≈ {expected_epochs} epochs"
            )
            print(
                f"[Client {self.cid}] Dataset: {dataset_size} samples, "
                f"Batch: {batch_size}, Updates/epoch: {updates_per_epoch:.2f}"
            )
        else:
            print(f"[Client {self.cid}] Training CoxPH for {self.config.num_epochs} epochs")

        proximal_mu = ins.config.get("proximal_mu", None)
        global_weights = None

        if ins.parameters.tensors:
            from flwr.common import parameters_to_ndarrays

            parameters = parameters_to_ndarrays(ins.parameters)
            params_list = self._reshape_flower_parameters(parameters)
            self.set_parameters(params_list)

            if proximal_mu is not None:
                global_weights = [torch.tensor(p).to(self.model.device) for p in params_list]

        run_id = os.environ.get("RUN_ID", "unknown")
        server_round = ins.config.get("server_round", "unknown")
        log_dir = os.path.join(self.config.experiment_dir, "client_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"client_{self.cid}_coxph_training.log")

        if server_round == 1 and run_id == "1":
            print(f"[Client {self.cid}] Client logs directory: {log_dir}")

        with open(log_file, "a") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"RUN {run_id} - ROUND {server_round} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 80}\n")

        print(f"[Client {self.cid}] === Run {run_id} - Round {server_round} ===")
        if proximal_mu is not None:
            print(f"[Client {self.cid}] Using FedProx with mu={proximal_mu}")

        self.model.fit(
            self.X_train,
            self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            verbose=True,
            client_id=self.cid,
            log_file=log_file,
            proximal_mu=proximal_mu,
            global_weights=global_weights,
        )

        self.model.reset_optimizer_state()

        from flwr.common import ndarrays_to_parameters

        parameters_proto = ndarrays_to_parameters(self._get_parameters_as_arrays())

        print(f"[Client {self.cid}] Local CoxPH training completed")

        return FitRes(
            status=Status(Code.OK, message="OK"),
            parameters=parameters_proto,
            num_examples=len(self.X_train),
            metrics={"client_id": self.cid},
        )

    def evaluate(self, ins):
        print(f"[Client {self.cid}] Starting CoxPH evaluation")

        if ins.parameters.tensors:
            from flwr.common import parameters_to_ndarrays

            parameters = parameters_to_ndarrays(ins.parameters)
            params_list = self._reshape_flower_parameters(parameters)
            self.set_parameters(params_list)

        self.model._estimate_baseline_hazard(self.X_train, self.y_train)

        metrics = evaluate_deepsurv(
            model=self.model,
            data={
                "X_test": self.X_test,
                "y_test": self.y_test,
                "y_train": self.y_train,
            },
            client_id=self.cid,
            config=self.config,
        )
        metrics["cid"] = self.cid
        metrics["client_name"] = self.name

        print(
            f"[Client {self.cid}] Evaluation finished → "
            f"C-index={metrics.get('C-index', np.nan):.4f}, "
            f"AUC={metrics.get('AUC', np.nan):.4f}, "
            f"IBS={metrics.get('IBS', np.nan):.4f}"
        )

        run_id = os.environ.get("RUN_ID", "unknown")
        server_round = ins.config.get("server_round", "unknown")
        csv_path = os.environ.get(
            "OUTPUT_CSV",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "results_randomness_exps",
                f"run_{run_id}.csv",
            ),
        )

        append_metrics_to_csv(
            csv_path,
            {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "round": server_round,
                "client_id": self.cid,
                "c_index": metrics["C-index"],
                "auc": metrics.get("AUC", float("nan")),
                "ibs": metrics.get("IBS", float("nan")),
            },
        )

        return EvaluateRes(
            status=Status(Code.OK, message="OK"),
            loss=0.0,
            num_examples=len(self.X_test),
            metrics=metrics,
        )

    def to_client(self):
        return self
