import numpy as np
import logging
from flwr.server.strategy import FedAvg, FedAdam, FedAdagrad, FedYogi, FedProx
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import copy
import flwr as fl
import pickle


class DeepSurvFedAvg(FedAvg):
    """
    FedAvg strategy for DeepSurv (PyTorch weight averaging).
    
    Uses standard federated averaging for neural network weights.
    Each client trains locally using local risk sets (biased approximation).
    """

    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate DeepSurv model weights from clients.
        
        Standard FedAvg: w_global = Σ(n_k/n_total * w_k)
        """
        if not results:
            return None, {}

        logger = logging.getLogger("main")
        logger.info(f"[Server] Round {server_round}: Aggregating DeepSurv weights from {len(results)} clients")

        # Extract weights and sample sizes
        weights_list = []
        num_examples_list = []

        for client_proxy, fit_res in results:
            # Convert bytes back to numpy arrays
            tensors = fit_res.parameters.tensors
            weights = [np.frombuffer(t, dtype=np.float32) for t in tensors]
            weights_list.append(weights)
            num_examples_list.append(fit_res.num_examples)

        # Compute weighted average
        total_examples = sum(num_examples_list)
        aggregated_weights = []

        for i in range(len(weights_list[0])):
            layer_weights = [
                weights[i] * (num_examples_list[j] / total_examples)
                for j, weights in enumerate(weights_list)
            ]
            aggregated_weights.append(np.sum(layer_weights, axis=0))

        # Convert back to bytes
        aggregated_tensors = [w.tobytes() for w in aggregated_weights]

        logger.info(
            f"[Server] Round {server_round}: aggregated {len(aggregated_weights)} weight tensors"
        )

        return (
            fl.common.Parameters(
                tensors=aggregated_tensors,
                tensor_type="numpy"
            ),
            {}
        )

    def evaluate(self, server_round, parameters, config=None):
        """Skip evaluation on round 0; use parent behavior otherwise."""
        if server_round == 0:
            logging.getLogger("main").info(
                "[Server] Skipping evaluation for round 0 (model not yet trained)."
            )
            return None
        return super().evaluate(server_round, parameters)

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure evaluation and pass round number to clients."""
        # Get the default evaluation configuration from parent
        eval_config = super().configure_evaluate(server_round, parameters, client_manager)
        
        # Inject server_round into each client's config
        if eval_config:
            updated_config = []
            for client, evaluate_ins in eval_config:
                # Add server_round to the config
                new_config = dict(evaluate_ins.config) if evaluate_ins.config else {}
                new_config["server_round"] = server_round
                
                # Create new EvaluateIns with updated config
                new_evaluate_ins = fl.common.EvaluateIns(
                    parameters=evaluate_ins.parameters,
                    config=new_config
                )
                updated_config.append((client, new_evaluate_ins))
            return updated_config
        
        return eval_config

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics and log per-client + aggregated results."""
        logger = logging.getLogger("main")

        if not results:
            logger.warning(f"[Server] No evaluation results in round {server_round}.")
            return None, {}

        # Log individual client metrics
        logger.info(f"──────────────────────────────────────────────")
        logger.info(f"[Server] Round {server_round} - Client Metrics:")
        logger.info(f"──────────────────────────────────────────────")

        metrics_dict = {}
        for client_proxy, evaluate_res in results:
            client_id = getattr(client_proxy, "cid", "Unknown")
            metrics = evaluate_res.metrics

            client_name = metrics.get("client_name", f"Client {client_id}")
            logger.info(
                f" → {client_name} (ID {client_id}): "
                f"C-index={metrics.get('C-index', np.nan):.4f}, "
                f"AUC={metrics.get('AUC', np.nan):.4f}, "
                f"IBS={metrics.get('IBS', np.nan):.4f}"
            )

            metrics_dict[client_id] = metrics

        logger.info(f"──────────────────────────────────────────────")

        # Aggregate across clients
        aggregated_metrics = aggregate_evaluate_metrics([(cid, m) for cid, m in metrics_dict.items()])

        logger.info(
            f"[Server] Round {server_round} - Aggregated Metrics → "
            f"C-index={aggregated_metrics.get('C-index', np.nan):.4f}, "
            f"AUC={aggregated_metrics.get('AUC', np.nan):.4f}, "
            f"IBS={aggregated_metrics.get('IBS', np.nan):.4f}"
        )

        return None, aggregated_metrics


class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy that logs and properly aggregates parameters."""

    # def aggregate_fit(self, server_round, results, failures):
    #     if not results:
    #         return None, {}

    #     logger = logging.getLogger("main")
    #     logger.info(f"[Server] Aggregating {len(results)} client results for round {server_round}...")

    #     # Convert all client Parameters to numpy arrays
    #     weights_results = []
    #     for _, fit_res in results:
    #         try:
    #             ndarrays = parameters_to_ndarrays(fit_res.parameters)
    #             weights_results.append(ndarrays)
    #         except Exception as e:
    #             logger.warning(f"[Server] Failed to parse client parameters: {e}")

    #     if not weights_results:
    #         logger.error("[Server] No valid weights received — skipping aggregation.")
    #         return None, {}

    #     # Average element-wise across clients
    #     aggregated_ndarrays = [
    #         np.mean([weights[i] for weights in weights_results], axis=0)
    #         for i in range(len(weights_results[0]))
    #     ]

    #     # Convert back to Flower Parameters object
    #     aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

    #     # Log aggregation statistics
    #     means = [np.mean(arr) for arr in aggregated_ndarrays]
    #     stds = [np.std(arr) for arr in aggregated_ndarrays]
    #     logger.info(
    #         f"[Server] Round {server_round}: aggregated weights "
    #         f"mean={means}, std={stds}"
    #     )

    #     return aggregated_parameters, {}

    def aggregate_fit(self, server_round, results, failures):
        logger = logging.getLogger("main")
        logger.info(f"[SERVER] Round {server_round}: Aggregating FIT results")

        if server_round != 1:
            return None, {}

        all_trees = []
        all_scores = []
        per_client_counts = []

        # 1. Collect all trees and scores
        for client_proxy, fit_res in results:
            payload = pickle.loads(fit_res.parameters.tensors[0])
            trees = payload["trees"]
            scores = payload["scores"]

            all_trees.extend(trees)
            all_scores.extend(scores)
            per_client_counts.append(len(trees))

            logger.info(
                f"[SERVER] Client {client_proxy.cid}: "
                f"sent {len(trees)} trees "
                f"(mean val C-index={np.mean(scores):.3f})"
            )

        all_scores = np.array(all_scores)
        all_scores = np.nan_to_num(all_scores, nan=0.5)
        all_scores[all_scores < 0.5] = 0.5

        # 2. Build sampling distribution (FedSurF++-C)
        probs = all_scores / all_scores.sum()

        N_global = self.num_trees_fed
        if N_global is None:
            raise ValueError("n_trees_federated must be set")

        # 3. Sample global forest
        selected_idx = np.random.choice(
            len(all_trees),
            size=N_global,
            replace=False,
            p=probs
        )

        global_forest = [copy.deepcopy(all_trees[i]) for i in selected_idx]

        logger.info(
            f"[SERVER] FedSurF++-C global forest built → "
            f"{len(global_forest)} trees "
            f"(from {len(all_trees)} total)"
        )

        return (
            fl.common.Parameters(
                tensors=[pickle.dumps(global_forest)],
                tensor_type="pickle"
            ),
            {}
        )


    def evaluate(self, server_round, parameters, config=None):
        """Skip evaluation on round 0; use parent behavior otherwise."""
        if server_round == 0:
            logging.getLogger("main").info(
                "[Server] Skipping evaluation for round 0 (model not yet trained)."
            )
            return None
        return super().evaluate(server_round, parameters)

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics and log per-client + aggregated results."""
        logger = logging.getLogger("main")

        if not results:
            logger.warning(f"[Server] No evaluation results in round {server_round}.")
            return None, {}

        # Log individual client metrics
        logger.info(f"──────────────────────────────────────────────")
        logger.info(f"[Server] Round {server_round} - Client Metrics:")
        logger.info(f"──────────────────────────────────────────────")

        metrics_dict = {}
        for client_proxy, evaluate_res in results:
            client_id = getattr(client_proxy, "cid", "Unknown")
            metrics = evaluate_res.metrics

            # Pretty print each client's metrics
            #client_name = getattr(client_proxy, "name", f"Client {client_id}")
            client_name = metrics.get("client_name", f"Client {client_id}")
            logger.info(
                f" → {client_name} (ID {client_id}): "
                f"C-index={metrics.get('C-index', np.nan):.4f}, "
                f"AUC={metrics.get('AUC', np.nan):.4f}, "
                f"IBS={metrics.get('IBS', np.nan):.4f}"
            )

            metrics_dict[client_id] = metrics

        logger.info(f"──────────────────────────────────────────────")

        # Aggregate across clients using the same function as before
        aggregated_metrics = aggregate_evaluate_metrics([(cid, m) for cid, m in metrics_dict.items()])

        logger.info(
            f"[Server] Round {server_round} - Aggregated Metrics → "
            f"C-index={aggregated_metrics.get('C-index', np.nan):.4f}, "
            f"AUC={aggregated_metrics.get('AUC', np.nan):.4f}, "
            f"IBS={aggregated_metrics.get('IBS', np.nan):.4f}"
        )

        return None, aggregated_metrics

# OUTSIDE THE CLASS
def aggregate_evaluate_metrics(metrics):
        """
        Aggregate client metrics (mean of each metric across clients).
        """
        results = {"C-index": [], "AUC": [], "IBS": []}
        for _, m in metrics:
            for k, v in m.items():
                if k in results and not np.isnan(v):
                    results[k].append(v)
        aggregated = {k: float(np.mean(v)) if v else np.nan for k, v in results.items()}
        logging.info(
            f"[Server] Round Metrics → "
            f"C-index={aggregated.get('C-index', np.nan):.4f}, "
            f"AUC={aggregated.get('AUC', np.nan):.4f}, "
            f"IBS={aggregated.get('IBS', np.nan):.4f}"
        )
        return aggregated




import numpy as np
import logging
from flwr.server.strategy import FedAvg
import pickle


# class FedSurvForest(FedAvg):
#     """
#     Federated Survival Forest (Simple version)

#     - Client-proportional sampling
#     - Tree-per-client budget correction
#     - Uniform tree sampling inside each client
#     """

#     def __init__(
#         self,
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#         num_trees_fed=None,     
#     ):
#         # store custom argument
#         self.num_trees_fed = num_trees_fed

#         # call FedAvg init with ONLY allowed arguments
#         super().__init__(
#             fraction_fit=fraction_fit,
#             fraction_evaluate=fraction_evaluate,
#             min_fit_clients=min_fit_clients,
#             min_evaluate_clients=min_evaluate_clients,
#             min_available_clients=min_available_clients,
#         )


#     def aggregate_fit(self, server_round, results, failures):
#         logger = logging.getLogger("main")

#         # ==============================================================
#         # ROUND 0: Ask clients to send event-time arrays only
#         # ==============================================================

#         if server_round == 1:
#             print(f"[DEBUG][Server] Round 1 aggregate_fit START — receiving event times from clients")

#             all_times = []

#             # --- DEBUG client event times received ---
#             for cid, (_, fit_res) in enumerate(results):
#                 arr = fl.common.parameters_to_ndarrays(fit_res.parameters)[0]

#                 print(f"[DEBUG][Server] Received times from client {cid}: count={len(arr)}")
#                 if len(arr) > 0:
#                     print(f"       min={arr.min():.3f}, max={arr.max():.3f}")
#                     print(f"       sample={np.sort(arr)[:5]}")

#                 all_times.extend(arr.tolist())

#             # Build global grid:
#             self.global_event_times = np.unique(all_times)
#             g = self.global_event_times

#             print(f"[DEBUG][Server] GLOBAL EVENT GRID computed: {len(g)} points")
#             if len(g) > 0:
#                 print(f"       min={g.min():.3f}, max={g.max():.3f}")
#                 print(f"       sample first 10: {g[:10]}")

#             import pickle

#             return (
#                 fl.common.Parameters(
#                     tensors=[
#                         pickle.dumps([]),               # empty forest placeholder
#                         pickle.dumps(self.global_event_times),  # real global grid
#                     ],
#                     tensor_type="pickle",
#                 ),
#                 {}
#             )


#         if server_round == 2:
#             print(f"[DEBUG][Server] aggregate_fit ROUND 2 — aggregating RSF trees")

#             if not results:
#                 logger.error("[Server] No results for round 2")
#                 return None, {}

#             # ---- Extract trees ----
#             client_trees = []
#             client_sizes = []

#             #for _, fit_res in results:
#             for client_proxy, fit_res in results:
#                 cid = client_proxy.cid
#                 import pickle
#                 trees = pickle.loads(fit_res.parameters.tensors[0])
#                 print(f"[SERVER-DEBUG] Client {cid} sent {len(trees)} trees")
#                 #print(f"               First tree time len = {len(trees[0].event_times_)}")

#                 client_trees.append(trees)
#                 client_sizes.append(fit_res.num_examples)

#             # ---- Probabilities ----
#             client_sizes = np.array(client_sizes, dtype=float)
#             total_samples = client_sizes.sum()
#             client_probs = client_sizes / total_samples

#             # ---- Determine N trees ----
#             total_available = sum(len(t) for t in client_trees)
#             NS = total_available if self.num_trees_fed is None else min(self.num_trees_fed, total_available)

#             # ---- Sample trees ----
#             sampled_clients = np.random.choice(len(client_trees), size=NS, p=client_probs)
#             trees_per_client = np.array([np.sum(sampled_clients == j) for j in range(len(client_trees))])

#             # Fix deficits
#             actual_available = np.array([len(t) for t in client_trees])
#             diff = actual_available - trees_per_client

#             import copy
#             while (diff < 0).any():
#                 deficit = np.random.choice(np.where(diff < 0)[0])
#                 surplus = np.random.choice(np.where(diff > 0)[0])
#                 trees_per_client[deficit] -= 1
#                 trees_per_client[surplus] += 1
#                 diff = actual_available - trees_per_client

#             # ---- Build global forest ----
#             federated_trees = []
#             for cid, n in enumerate(trees_per_client):
#                 pool = client_trees[cid]
#                 if n > 0:
#                     chosen = np.random.choice(pool, size=n, replace=False)
#                     federated_trees.extend([copy.deepcopy(t) for t in chosen])

#             # ---- Send aggregated trees ----
#             import pickle
#             return (
#                 fl.common.Parameters(
#                     tensors=[
#                         pickle.dumps(federated_trees),
#                         pickle.dumps(self.global_event_times),
#                     ],
#                     tensor_type="pickle",
#                 ),
#                 {}
#             )

#         # ROUND ≥3 (Flower): No training — only evaluation happens
#         print(f"[DEBUG][Server] aggregate_fit ROUND {server_round} — nothing to aggregate")
#         return None, {}


#     def configure_fit(self, server_round, parameters, client_manager):
#         # I HAVE TO HANDLE ROUND 0, BC FLOWER INTERNALLY ALWAYS CALLS ROUND 0
#         if server_round == 0:
#             print("[DEBUG][Server] configure_fit: ROUND 0 (Flower init) — do NOTHING")
#             return []    # no clients should run in round 0

#         # ROUND 1 → clients send ONLY event times
#         if server_round == 1:
#             print("[DEBUG][Server] configure_fit: ROUND 1 request event times")

#              # Sample clients through Flower
#             # sample = client_manager.sample(
#             #     num_clients=self.min_fit_clients,
#             #     min_num_clients=self.min_fit_clients,
#             # )
#             sample = list(client_manager.all().values())

#             # Return list of (client_proxy, FitIns)
#             fit_ins = fl.common.FitIns(
#                 parameters=fl.common.Parameters(tensors=[], tensor_type=""),
#                 config={"send_event_times": True},
#             )

#             return [(client, fit_ins) for client in sample]

#         # ROUND 2 → local training (normal RSF training)
#         if server_round == 2:
#             print("[DEBUG][Server] configure_fit: ROUND 2 local RSF training")
#             # Sample clients again
#             sample = list(client_manager.all().values())

#             # Send the global event grid to clients BEFORE they train
#             fit_ins = fl.common.FitIns(
#                 parameters=fl.common.Parameters(
#                     tensors=[pickle.dumps(self.global_event_times)],
#                     tensor_type="pickle",
#                 ),
#                 config={}
#             )

#             # Return instructions for each selected client
#             return [(client, fit_ins) for client in sample]


#         # ROUND ≥3 → NO MORE TRAINING
#         print("[DEBUG][Server] configure_fit: ROUND ≥3 no more training")
#         return []

#     def configure_evaluate(self, server_round, parameters, client_manager):
#         # Disable evaluation until after global forest exists
#         if server_round <= 2:
#             print("[DEBUG][Server] configure_evaluate: disabled")
#             return []
#         print("[DEBUG][Server] configure_evaluate: enabled")
#         return super().configure_evaluate(server_round, parameters, client_manager)

#     def evaluate(self, server_round, parameters):
#         # Disable server-side evaluation in rounds 0 and 1
#         if server_round < 2:
#             return None  # no server evaluation
#         return super().evaluate(server_round, parameters)

#     def initialize_parameters(self, client_manager):
#         # Disable default "initial model request"
#         print("[DEBUG][Server] initialize_parameters() called → returning EMPTY parameters")
#         import pickle
#         return fl.common.Parameters(
#             tensors=[], tensor_type="pickle"
#         )

# Training_Modes/Federated_Learning/strategies.py (FedSurvForest)
import flwr as fl
import numpy as np
import pickle
import copy
import logging

class FedSurvForest(fl.server.strategy.FedAvg):
    """
    Correct FedSurF implementation for Flower:
      - Round 1: clients train RSF and return trees
      - Server: weighted random sampling of trees
      - Round 2+: evaluation only
    """

    def __init__(
        self,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        num_trees_fed=None,
    ):

        self.num_trees_fed = num_trees_fed

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )


    # # ROUND 1: aggregate RSF trees
    # def aggregate_fit(self, server_round, results, failures):
    #     '''
    #     This method did get the local trees per client randomly, but weighted per its data size.
    #     '''
    #     logger = logging.getLogger("main")
    #     logger.info(f"[SERVER] Round {server_round}: Aggregating FIT results")
    #     logger.info(f"  {len(results)} clients returned")

    #     if server_round == 1:
    #         logger.info("[Server] ROUND 1: Aggregating local RSF forests")

    #         # collect trees + sample counts
    #         all_trees = []
    #         client_sizes = []

    #         for client, fit_res in results:
    #             trees = pickle.loads(fit_res.parameters.tensors[0])
    #             all_trees.append(trees)
    #             client_sizes.append(fit_res.num_examples)

    #         client_sizes = np.array(client_sizes)
    #         total_samples = client_sizes.sum()
    #         probs = client_sizes / total_samples

    #         # number of global trees
    #         total_local = sum(len(tlist) for tlist in all_trees)
    #         NS = total_local if self.num_trees_fed is None else min(self.num_trees_fed, total_local)

    #         # sample client assignments
    #         sampled_clients = np.random.choice(
    #             len(all_trees), size=NS, p=probs
    #         )

    #         trees_per_client = [
    #             np.sum(sampled_clients == cid)
    #             for cid in range(len(all_trees))
    #         ]

    #         # ensure no client is asked for more trees than it has
    #         actual_avail = np.array([len(tl) for tl in all_trees])
    #         diff = actual_avail - trees_per_client
    #         while (diff < 0).any():
    #             deficit = np.random.choice(np.where(diff < 0)[0])
    #             surplus = np.random.choice(np.where(diff > 0)[0])
    #             trees_per_client[deficit] -= 1
    #             trees_per_client[surplus] += 1
    #             diff = actual_avail - trees_per_client

    #         # -----------------------------------------------------
    #         # build global forest
    #         # -----------------------------------------------------
    #         global_forest = []
    #         for cid, n in enumerate(trees_per_client):
    #             pool = all_trees[cid]
    #             chosen = np.random.choice(pool, size=n, replace=False)
    #             global_forest.extend([copy.deepcopy(t) for t in chosen])

    #         logger.info(f"[Server] Global federated forest built: {len(global_forest)} trees")

    #         # SEND to clients in next round
    #         return (
    #             fl.common.Parameters(
    #                 tensors=[pickle.dumps(global_forest)],
    #                 tensor_type="pickle"
    #             ),
    #             {}
    #         )

    #     # ALL OTHER ROUNDS: nothing to aggregate
    #     return None, {}

    # def aggregate_fit(self, server_round, results, failures):
    #     '''
    #     This method was based on tree client selection by the FedSurF-C metric on the validation set.
    #     '''
    #     logger = logging.getLogger("main")
    #     logger.info(f"[SERVER] Round {server_round}: Aggregating FIT results")
    #     logger.info(f"  {len(results)} clients returned")

    #     # FedSurF-C: aggregation only happens in round 1
    #     if server_round == 1:
    #         logger.info("[Server] ROUND 1: Aggregating FedSurF-C selected trees")

    #         global_forest = []
    #         client_contributions = []

    #         for client, fit_res in results:
    #             trees = pickle.loads(fit_res.parameters.tensors[0])
    #             global_forest.extend([copy.deepcopy(t) for t in trees])
    #             client_contributions.append(len(trees))

    #         logger.info(
    #             "[Server] FedSurF-C global forest built → "
    #             f"{len(global_forest)} trees | "
    #             f"per-client contributions={client_contributions}"
    #         )

    #         return (
    #             fl.common.Parameters(
    #                 tensors=[pickle.dumps(global_forest)],
    #                 tensor_type="pickle"
    #             ),
    #             {}
    #         )

    #     # ALL OTHER ROUNDS: nothing to aggregate
    #     return None, {}

    def aggregate_fit(self, server_round, results, failures):
        """
        GLOBAL C-index–based tree selection.
        Each client trains n_trees_local trees.
        Server selects exactly n_trees_federated trees globally.
        """

        logger = logging.getLogger("main")
        logger.info(f"[SERVER] Round {server_round}: Aggregating FIT results")

        # Only aggregate in round 1
        if server_round != 1:
            return None, {}

        all_trees = []
        all_scores = []
        per_client_counts = {}

        # ---------------------------------------------------------
        # 1. Collect all trees + C-index scores from clients
        # ---------------------------------------------------------
        for client_proxy, fit_res in results:
            cid = getattr(client_proxy, "cid", "unknown")

            payload = pickle.loads(fit_res.parameters.tensors[0])

            # ---- STRICT payload validation ----
            if not isinstance(payload, dict):
                raise ValueError(f"[SERVER] Client {cid} sent non-dict payload")

            if "trees" not in payload or "scores" not in payload:
                raise ValueError(
                    f"[SERVER] Client {cid} payload keys invalid: {payload.keys()}"
                )

            trees = list(payload["trees"])
            scores = np.asarray(payload["scores"])

            scores = np.atleast_1d(scores)

            if len(trees) != len(scores):
                raise ValueError(
                    f"[SERVER] Client {cid}: {len(trees)} trees but {len(scores)} scores"
                )

            all_trees.extend(trees)
            all_scores.extend(scores.tolist())
            per_client_counts[cid] = len(trees)

            logger.info(
                f"[SERVER] Client {cid}: "
                f"trees={len(trees)}, mean C-index={scores.mean():.3f}"
            )

        logger.info(
            "[SERVER] Received trees per client: "
            + " | ".join(f"Client {cid}: {n}" for cid, n in per_client_counts.items())
        )

        all_scores = np.asarray(all_scores)

        # ---------------------------------------------------------
        # 2. Numerical safety
        # ---------------------------------------------------------
        all_scores = np.nan_to_num(all_scores, nan=0.5)
        all_scores[all_scores < 0.5] = 0.5

        # ---------------------------------------------------------
        # 3. Global probabilistic selection (C-index weighted)
        # ---------------------------------------------------------
        probs = all_scores / all_scores.sum()

        N_global = self.num_trees_fed  # = config.n_trees_federated

        if N_global > len(all_trees):
            logger.warning(
                f"[SERVER] Requested {N_global} trees, but only "
                f"{len(all_trees)} available. Clipping."
            )
            N_global = len(all_trees)

        selected_idx = np.random.choice(
            len(all_trees),
            size=N_global,
            replace=False,
            p=probs
        )

        global_forest = [copy.deepcopy(all_trees[i]) for i in selected_idx]

        logger.info(
            f"[SERVER] Global forest selected: {len(global_forest)} trees "
            f"(from {len(all_trees)} total)"
        )

        # ---------------------------------------------------------
        # 4. Send global forest to clients
        # ---------------------------------------------------------
        return (
            fl.common.Parameters(
                tensors=[pickle.dumps(global_forest)],
                tensor_type="pickle"
            ),
            {}
        )






    # configure round-by-round client instructions
    def configure_fit(self, server_round, parameters, client_manager):

        # ROUND 1: train local RSF
        if server_round == 1:
            sample = list(client_manager.all().values())
            fit_ins = fl.common.FitIns(
                parameters=fl.common.Parameters(tensors=[], tensor_type=""),
                config={}
            )
            return [(client, fit_ins) for client in sample]

        # ROUND >= 2: no training
        return []

    def configure_evaluate(self, server_round, parameters, client_manager):
        if server_round <= 1:
            return []
        return super().configure_evaluate(server_round, parameters, client_manager)


    def aggregate_evaluate(self, server_round, results, failures):
        logger = logging.getLogger("main")
        logger.info(f"[SERVER] Round {server_round}: Aggregating EVALUATION")

        for _, eval_res in results:
            cid = eval_res.metrics["cid"]
            m = eval_res.metrics
            logger.info(
                f"[FedSurF][Client {cid}] "
                f"C-index={m['C-index']:.4f} | "
                f"AUC={m['AUC']:.4f} | "
                f"IBS={m['IBS']:.4f}"
            )

        # Keep original behavior
        return super().aggregate_evaluate(server_round, results, failures)





def get_strategy(strategy_name: str, **kwargs):
    """Return the selected FL strategy (CustomFedAvg by default)."""
    strategies = {
        "FedAvg": CustomFedAvg,
        "FedAdam": FedAdam,
        "FedAdagrad": FedAdagrad,
        "FedYogi": FedYogi,
        "FedProx": FedProx,
        "FedSurvForest": FedSurvForest,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unsupported strategy: {strategy_name}. "
            f"Supported: {list(strategies.keys())}"
        )

    logger = logging.getLogger("main")
    logger.info(f"[Strategy] Selected strategy: {strategy_name}")

    return strategies[strategy_name](**kwargs)
