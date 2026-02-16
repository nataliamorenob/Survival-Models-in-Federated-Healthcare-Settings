# Training_Modes/Federated_Learning/clientDeepSurv.py
"""
Federated DeepSurv Client using Local Risk Set Approximation

WARNING: This implementation uses local risk sets for Cox loss computation,
which introduces bias compared to centralized training. This is done for
privacy preservation but sacrifices mathematical correctness.

Reference: Local risk set approximation with FedAvg aggregation.
"""

import flwr as fl
import numpy as np
import torch
from flwr.common import FitRes, EvaluateRes, GetParametersRes, Parameters, Status, Code, NDArrays
from utils import evaluate_deepsurv
import os
from datetime import datetime
from Exps_runs_randomness.utils_results import append_metrics_to_csv


class FederatedDeepSurvClient(fl.client.Client):
    """
    Federated Learning client for DeepSurv using local risk set approximation.
    
    Each hospital trains DeepSurv locally using ONLY local patient data to
    compute risk sets. Model weights are aggregated via FedAvg.
    
    NOTE: This approach is biased compared to centralized Cox models because
    risk sets R(t_i) should contain ALL patients across hospitals, not just
    local patients.
    """

    def __init__(self, cid, name, model, config, dataloaders):
        """
        Initialize Federated DeepSurv Client.
        
        Args:
            cid: Client ID (hospital/center index)
            name: Client name (hospital/center name)
            model: DeepSurv model instance
            config: Configuration object
            dataloaders: Dictionary containing train/val/test data
        """
        self.cid = cid
        self.name = name
        self.model = model
        self.config = config

        # Extract data for this center
        center = list(dataloaders.keys())[0]
        data = dataloaders[center]

        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        self.X_test = data["X_test"]
        self.y_test = data["y_test"]

    def get_parameters(self, ins=None):
        """
        Extract model weights and return as GetParametersRes.
        
        Returns:
            GetParametersRes with serialized parameters
        """
        # Extract weights as numpy arrays
        parameters = self._get_parameters_as_arrays()
        
        # Convert to bytes for transmission
        tensors = [param.tobytes() for param in parameters]
        
        return GetParametersRes(
            status=Status(Code.OK, message="OK"),
            parameters=Parameters(
                tensors=tensors,
                tensor_type="numpy"
            )
        )

    def _get_parameters_as_arrays(self):
        """Helper method to get parameters as list of numpy arrays."""
        return [
            param.cpu().detach().numpy() 
            for param in self.model.network.state_dict().values()
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set model weights from numpy arrays.
        
        Args:
            parameters: List of numpy arrays representing model parameters
        """
        state_dict = self.model.network.state_dict()
        for param_tensor, new_value in zip(state_dict.values(), parameters):
            param_tensor.copy_(torch.tensor(new_value))

    def fit(self, ins):
        """
        Local training using LOCAL risk sets (biased approximation).
        
        WARNING: This computes Cox loss using only local patients' risk sets,
        which is mathematically incorrect but necessary for privacy.
        """
        print(f"[Client {self.cid}] Training DeepSurv for {self.config.num_epochs} epochs")

        # Set global weights if provided
        if ins.parameters.tensors:
            # Server sent global weights - load them
            parameters = [
                np.frombuffer(tensor, dtype=np.float32) 
                for tensor in ins.parameters.tensors
            ]
            # Reshape parameters to match model architecture
            param_idx = 0
            state_dict = self.model.network.state_dict()
            params_list = []
            for key, param in state_dict.items():
                param_shape = param.shape
                param_size = param.numel()
                params_list.append(
                    parameters[param_idx][:param_size].reshape(param_shape)
                )
                param_idx += 1
            self.set_parameters(params_list)

        # Train locally using LOCAL risk sets with validation for early stopping
        # NOTE: This is the biased approximation - risk sets only contain local patients
        self.model.fit(
            self.X_train, 
            self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            verbose=False  # Reduce output in federated setting
        )

        # Clear optimizer state to reduce memory usage between rounds
        self.model.reset_optimizer_state()

        # Extract updated weights as numpy arrays
        parameters = self._get_parameters_as_arrays()

        # Convert to bytes for transmission
        tensors = [param.tobytes() for param in parameters]

        print(f"[Client {self.cid}] Local training completed")

        return FitRes(
            status=Status(Code.OK, message="OK"),
            parameters=Parameters(
                tensors=tensors,
                tensor_type="numpy"
            ),
            num_examples=len(self.X_train),
            metrics={"client_id": self.cid}
        )

    def evaluate(self, ins):
        """
        Evaluate global model on local test data.
        
        Args:
            ins: Evaluation instructions from server containing global weights
            
        Returns:
            Evaluation results with metrics
        """
        print(f"[Client {self.cid}] Starting evaluation")

        # Load global weights sent by server
        if ins.parameters.tensors:
            parameters = [
                np.frombuffer(tensor, dtype=np.float32) 
                for tensor in ins.parameters.tensors
            ]
            # Reshape parameters to match model architecture
            param_idx = 0
            state_dict = self.model.network.state_dict()
            params_list = []
            for key, param in state_dict.items():
                param_shape = param.shape
                param_size = param.numel()
                params_list.append(
                    parameters[param_idx][:param_size].reshape(param_shape)
                )
                param_idx += 1
            self.set_parameters(params_list)

        # Re-estimate baseline hazard using LOCAL training data
        # This is necessary because baseline hazard is non-parametric
        self.model._estimate_baseline_hazard(self.X_train, self.y_train)

        # Evaluate on local test set
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

        # Save metrics to CSV for randomness experiments
        run_id = os.environ.get("RUN_ID", "unknown")
        server_round = ins.config.get("server_round", "unknown")
        csv_path = os.environ.get(
            "OUTPUT_CSV",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "results_randomness_exps",
                f"run_{run_id}.csv"
            )
        )

        append_metrics_to_csv(
            csv_path,
            {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "round": server_round,
                "client_id": self.cid,
                "c_index": metrics["C-index"],
                "auc": metrics.get("AUC", float('nan')),
                "ibs": metrics.get("IBS", float('nan')),
            }
        )

        return EvaluateRes(
            status=Status(Code.OK, message="OK"),
            loss=0.0,  # We don't compute loss during evaluation
            num_examples=len(self.X_test),
            metrics=metrics
        )

    def to_client(self):
        """Convert to Flower Client for simulation."""
        return self
