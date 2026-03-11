# Training_Modes/Federated_Learning/clientRSFFedSurF.py
"""
FedSurF++ Client Implementation (PAPER-EXACT)
Implements the exact three-phase algorithm from the paper.
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
    FedSurF++ Client (EXACT paper implementation):
    
    Round 1 (Metadata): Send Tk (num trees) and Nk (dataset size)
    Round 2 (Tree Sampling): 
        - Receive T'_k from server
        - Evaluate trees with C-Index
        - Sample T'_k trees proportionally
        - Send selected trees
    Round 3+ (Evaluation): Evaluate global forest
    """

    def __init__(self, cid, name, model, config, dataloaders):
        """Initialize FedSurF++ client."""
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
        
        # Store trained trees and scores for round 2
        self.local_trees = None
        self.tree_scores = None

    def fit(self, ins):
        """
        FedSurF++ fit() handles TWO different rounds:
        
        ROUND 1: Local Training + Metadata
            - Train local RSF with Tk trees
            - Send Tk and Nk to server
        
        ROUND 2: Tree Sampling + Transfer
            - Receive T'_k from server
            - Evaluate trees with C-Index
            - Sample T'_k trees proportional to C-Index
            - Send selected trees to server
        """
        # Determine which round by checking config
        round_type = ins.config.get("round_type", "metadata")
        
        if round_type == "metadata":
            return self._fit_round1_metadata()
        elif round_type == "tree_sampling":
            return self._fit_round2_sampling(ins)
        else:
            raise ValueError(f"Unknown round_type: {round_type}")
    
    def _fit_round1_metadata(self):
        """
        ROUND 1: Local Training + Tree Evaluation + Send Metadata
        
        Algorithm steps (from pseudocode):
        1. Train local RSF → get Tk trees
        2. Send Tk (and Nk) to server for tree assignment
        3. Compute Metric_j for each tree j ∈ Mk (Tree sampling section)
           → This happens BEFORE receiving T'_k (computed while waiting)
        """
        self.logger.info(
            f"[Client {self.cid}][Round 1] Local Training: "
            f"n_trees_local={self.config.n_trees_local}"
        )

        # ================================================================
        # PHASE 1: Local Training (Paper: lines 1-3)
        # ================================================================
        self.model.fit(self.X_train, self.y_train)
        self.local_trees = self.model.estimators_
        Tk = len(self.local_trees)
        Nk = len(self.X_train)

        self.logger.info(
            f"[Client {self.cid}][Round 1] Trained: "
            f"Tk={Tk} trees, Nk={Nk} samples"
        )

        # ================================================================
        # TREE SAMPLING PREPARATION (Paper: "for j = 1 to Tk do")
        # Compute Metric_j for each tree BEFORE receiving T'_k
        # This follows the pseudocode structure exactly
        # ================================================================
        self.logger.info(
            f"[Client {self.cid}][Round 1] Computing C-Index for all {Tk} trees..."
        )
        
        scores = []
        for tree_idx, tree in enumerate(self.local_trees):
            try:
                surv_fns = tree.predict_survival_function(self.X_val)
                risks = np.array([
                    -np.log(fn.y[-1] + 1e-8) for fn in surv_fns
                ])
                c_index = concordance_index_censored(
                    self.y_val["event"],
                    self.y_val["time"],
                    risks
                )[0]
            except Exception as e:
                c_index = 0.5

            scores.append(c_index)

        self.tree_scores = np.array(scores)
        self.tree_scores = np.nan_to_num(self.tree_scores, nan=0.5)
        self.tree_scores[self.tree_scores < 0.5] = 0.5

        self.logger.info(
            f"[Client {self.cid}][Round 1] Tree metrics computed: "
            f"mean={self.tree_scores.mean():.4f}, "
            f"range=[{self.tree_scores.min():.4f}, {self.tree_scores.max():.4f}]"
        )

        # ================================================================
        # Send metadata (Tk, Nk) to server
        # ================================================================
        metadata = {
            "Tk": Tk,
            "Nk": Nk,
        }

        return FitRes(
            status=Status(Code.OK, message="Metadata sent"),
            parameters=Parameters(
                tensors=[pickle.dumps(metadata)],
                tensor_type="pickle"
            ),
            num_examples=Nk,
            metrics={"Tk": Tk, "Nk": Nk, "mean_c_index": float(self.tree_scores.mean())}
        )
    
    def _fit_round2_sampling(self, ins):
        """
        ROUND 2: Tree Selection + Transfer
        
        Algorithm steps (Paper pseudocode):
        1. Receive T'_k from server
        2. Select T'_k trees using probabilities proportional to Metric_j
           (Metrics already computed in Round 1)
        3. Send selected trees to server
        
        NOTE: In Flower simulation, clients are recreated each round,
        so we need to retrain if local_trees is None.
        """
        # ================================================================
        # Handle client state recreation (Flower simulation issue)
        # ================================================================
        if self.local_trees is None or self.tree_scores is None:
            self.logger.info(
                f"[Client {self.cid}][Round 2] Client state lost - retraining model"
            )
            # Retrain model and recompute scores
            self.model.fit(self.X_train, self.y_train)
            self.local_trees = self.model.estimators_
            
            # Recompute tree scores
            scores = []
            for tree in self.local_trees:
                try:
                    surv_fns = tree.predict_survival_function(self.X_val)
                    risks = np.array([-np.log(fn.y[-1] + 1e-8) for fn in surv_fns])
                    c_index = concordance_index_censored(
                        self.y_val["event"], self.y_val["time"], risks
                    )[0]
                except Exception:
                    c_index = 0.5
                scores.append(c_index)
            
            self.tree_scores = np.array(scores)
            self.tree_scores = np.nan_to_num(self.tree_scores, nan=0.5)
            self.tree_scores[self.tree_scores < 0.5] = 0.5
        
        # ================================================================
        # Receive T'_k from server (Paper: "Receive T'_k")
        # ================================================================
        Tk_prime = ins.config.get("Tk_prime", 0)
        
        self.logger.info(
            f"[Client {self.cid}][Round 2] Received Tree Assignment: "
            f"T'_k={Tk_prime} (out of {len(self.local_trees)} local trees)"
        )
        
        if Tk_prime == 0:
            # Client not selected - send empty
            self.logger.info(f"[Client {self.cid}][Round 2] Not selected (T'_k=0)")
            return FitRes(
                status=Status(Code.OK, message="Not selected"),
                parameters=Parameters(tensors=[pickle.dumps([])], tensor_type="pickle"),
                num_examples=0,
                metrics={"Tk_prime": 0}
            )

        # ================================================================
        # PHASE 3: Tree Sampling (Paper: "Select T'_k trees using 
        #          probabilities proportional to Metric_j")
        # Use pre-computed (or freshly computed) metrics
        # ================================================================
        
        # Build probability distribution: p_j = Metric_j / Σ(Metric_j)
        sampling_probs = self.tree_scores / self.tree_scores.sum()
        
        # Sample T'_k trees WITHOUT replacement
        Tk_prime_capped = min(Tk_prime, len(self.local_trees))
        
        selected_indices = np.random.choice(
            len(self.local_trees),
            size=Tk_prime_capped,
            replace=False,
            p=sampling_probs
        )
        
        selected_trees = [self.local_trees[i] for i in selected_indices]
        selected_scores = self.tree_scores[selected_indices]

        self.logger.info(
            f"[Client {self.cid}][Round 2] Selected {len(selected_trees)} trees: "
            f"C-Index range=[{selected_scores.min():.4f}, {selected_scores.max():.4f}]"
        )

        # ================================================================
        # Send selected trees to server (Paper: "Send selected trees")
        # ================================================================
        return FitRes(
            status=Status(Code.OK, message="Trees sent"),
            parameters=Parameters(
                tensors=[pickle.dumps(selected_trees)],
                tensor_type="pickle"
            ),
            num_examples=len(self.X_train),
            metrics={
                "Tk_prime": Tk_prime_capped,
                "mean_selected_c_index": float(selected_scores.mean())
            }
        )

    def evaluate(self, ins):
        """
        FedSurF++ Evaluation Phase (Round 3+):
        
        Load global federated forest from server and evaluate on test set.
        Uses the paper-style evaluation with C-Index, AUC, and IBS.
        """
        # Get the current round number from config
        server_round = ins.config.get("server_round", 3)
        
        self.logger.info(
            f"[Client {self.cid}][Round {server_round}] FedSurF++ Evaluation"
        )

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
            f"[Client {self.cid}][Round {server_round}] Loaded global forest: "
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
            f"[Client {self.cid}][Round {server_round}] Test Metrics → "
            f"C-index={metrics.get('C-index', np.nan):.4f}, "
            f"AUC={metrics.get('AUC', np.nan):.4f}, "
            f"IBS={metrics.get('IBS', np.nan):.4f}"
        )

        # ================================================================
        # Save metrics to CSV (for experiment tracking)
        # ================================================================
        try:
            append_metrics_to_csv(
                round_num=server_round,
                client_id=self.cid,
                metrics=metrics,
                config=self.config
            )
            self.logger.info(
                f"[Client {self.cid}][Round {server_round}] Metrics saved to CSV"
            )
        except Exception as e:
            self.logger.warning(
                f"[Client {self.cid}][Round {server_round}] "
                f"Failed to save metrics to CSV: {e}"
            )

        return EvaluateRes(
            status=Status(Code.OK, message="OK"),
            loss=float(metrics.get("IBS", 1.0)),  # Use IBS as loss
            num_examples=len(self.X_test),
            metrics=metrics
        )
