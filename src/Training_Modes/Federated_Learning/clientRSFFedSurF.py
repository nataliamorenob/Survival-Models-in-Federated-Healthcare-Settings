# Training_Modes/Federated_Learning/clientRSFFedSurF.py
"""
FedSurF++ Client Implementation
Implements the client-side logic for FedSurF++ with C-Index based tree sampling.
"""

import flwr as fl
import pickle
import numpy as np
from flwr.common import FitRes, EvaluateRes, Parameters, Status, Code
from utils import evaluate_rsf
from sksurv.metrics import concordance_index_censored
from Exps_runs_randomness.utils_results import append_metrics_to_csv
import logging


class FederatedRSFFedSurFClient(fl.client.Client):
    """
    FedSurF++ Client following the algorithm from the paper:
    
    Round 1 (Training):
        1. Local Training: Train local RSF with n_trees_local trees
        2. Tree Evaluation: Compute C-Index for each tree on validation set
        3. Send to Server: Send ALL trees + C-Index scores
    
    Round 2+ (Evaluation):
        - Load global forest from server
        - Evaluate on test set
    """

    def __init__(self, cid, name, model, config, dataloaders):
        """
        Initialize FedSurF++ client.
        
        Args:
            cid: Client ID
            name: Client name (e.g., "center_0")
            model: RSFFedSurFPlus model instance
            config: Configuration object
            dataloaders: Dictionary containing train/val/test data
        """
        self.cid = cid
        self.name = name
        self.model = model
        self.config = config
        self.logger = logging.getLogger("main")

        # Extract data from dataloaders
        center = list(dataloaders.keys())[0]
        data = dataloaders[center]

        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        self.X_test = data["X_test"]
        self.y_test = data["y_test"]

    def fit(self, ins):
        """
        FedSurF++ Training Phase (Round 1):
        
        Step 1: Local Training
            - Train local RSF with n_trees_local trees
        
        Step 2: Tree Evaluation (C-Index on validation set)
            - Evaluate each tree individually on validation data
            - Compute C-Index as the tree quality metric
        
        Step 3: Send Results
            - Send ALL trees + C-Index scores to server
            - Server will handle tree assignment and sampling
        """
        self.logger.info(
            f"[Client {self.cid}] FedSurF++ Training: "
            f"n_trees_local={self.config.n_trees_local}"
        )

        # ================================================================
        # STEP 1: Local Training
        # ================================================================
        self.model.fit(self.X_train, self.y_train)
        trees = self.model.estimators_
        n_trees = len(trees)

        self.logger.info(
            f"[Client {self.cid}] Local RSF trained: {n_trees} trees"
        )

        # ================================================================
        # STEP 2: Tree Evaluation (C-Index per tree)
        # ================================================================
        # Following FedSurF++ paper: evaluate each tree on validation set
        # using C-Index (Concordance Index without IPCW weighting)
        
        scores = []
        for tree_idx, tree in enumerate(trees):
            try:
                # Get survival predictions from this single tree
                surv_fns = tree.predict_survival_function(self.X_val)
                
                # Convert survival curves to risk scores
                # Risk = -log(S(t_max)) where S(t_max) is survival at last observed time
                risks = np.array([
                    -np.log(fn.y[-1] + 1e-8) for fn in surv_fns
                ])

                # Compute C-Index for this tree
                c_index = concordance_index_censored(
                    self.y_val["event"],
                    self.y_val["time"],
                    risks
                )[0]
                
            except Exception as e:
                # Fallback to 0.5 (random prediction) if tree evaluation fails
                self.logger.warning(
                    f"[Client {self.cid}] Tree {tree_idx} evaluation failed: {e}"
                )
                c_index = 0.5

            scores.append(c_index)

        scores = np.array(scores)
        
        # Numerical safety: handle NaN and invalid scores
        scores = np.nan_to_num(scores, nan=0.5)
        scores[scores < 0.5] = 0.5  # C-Index should be at least 0.5 (random guess)

        self.logger.info(
            f"[Client {self.cid}] Tree C-Index scores: "
            f"mean={scores.mean():.4f}, min={scores.min():.4f}, max={scores.max():.4f}"
        )

        # ================================================================
        # STEP 3: Send ALL trees + scores to server
        # ================================================================
        # Server will perform:
        #   - Tree Assignment: decide how many trees from this client
        #   - Tree Sampling: sample trees proportional to C-Index
        
        payload = {
            "trees": trees,
            "scores": scores,
            "n_samples": len(self.X_train),  # For weighted tree assignment
        }

        return FitRes(
            status=Status(Code.OK, message="OK"),
            parameters=Parameters(
                tensors=[pickle.dumps(payload)],
                tensor_type="pickle"
            ),
            num_examples=len(self.X_train),
            metrics={"n_trees": n_trees, "mean_c_index": float(scores.mean())}
        )

    def evaluate(self, ins):
        """
        FedSurF++ Evaluation Phase (Round 2+):
        
        Load global federated forest from server and evaluate on test set.
        Uses the paper-style evaluation with C-Index, AUC, and IBS.
        """
        self.logger.info(f"[Client {self.cid}] FedSurF++ Evaluation")

        # ================================================================
        # Load global forest from server
        # ================================================================
        global_forest = pickle.loads(ins.parameters.tensors[0])

        # Initialize model with global forest
        self.model.set_trees(
            global_forest,
            n_features=self.X_train.shape[1],
            init_times=self.config.union_time_grid
        )

        self.logger.info(
            f"[Client {self.cid}] Loaded global forest: "
            f"{len(global_forest)} trees"
        )

        # ================================================================
        # Evaluate on test set using paper-style metrics
        # ================================================================
        test_data = {
            "X_test": self.X_test,
            "y_test": self.y_test,
            "y_train": self.y_train,
        }

        metrics = evaluate_rsf(
            model=self.model,
            data=test_data,
            client_id=self.cid,
            config=self.config
        )

        # Add client metadata
        metrics["cid"] = self.cid
        metrics["client_name"] = self.name

        self.logger.info(
            f"[Client {self.cid}] Test Metrics → "
            f"C-index={metrics.get('C-index', np.nan):.4f}, "
            f"AUC={metrics.get('AUC', np.nan):.4f}, "
            f"IBS={metrics.get('IBS', np.nan):.4f}"
        )

        # ================================================================
        # Save metrics to CSV (for experiment tracking)
        # ================================================================
        try:
            append_metrics_to_csv(
                round_num=2,  # Evaluation happens in round 2+
                client_id=self.cid,
                metrics=metrics,
                config=self.config
            )
        except Exception as e:
            self.logger.warning(
                f"[Client {self.cid}] Failed to save metrics to CSV: {e}"
            )

        return EvaluateRes(
            status=Status(Code.OK, message="OK"),
            loss=float(metrics.get("IBS", 1.0)),  # Use IBS as loss
            num_examples=len(self.X_test),
            metrics=metrics
        )
