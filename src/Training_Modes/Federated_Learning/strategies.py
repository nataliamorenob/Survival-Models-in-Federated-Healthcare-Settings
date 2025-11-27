import numpy as np
import logging
from flwr.server.strategy import FedAvg, FedAdam, FedAdagrad, FedYogi, FedProx
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy that logs and properly aggregates parameters."""

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        logger = logging.getLogger("main")
        logger.info(f"[Server] Aggregating {len(results)} client results for round {server_round}...")

        # Convert all client Parameters to numpy arrays
        weights_results = []
        for _, fit_res in results:
            try:
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                weights_results.append(ndarrays)
            except Exception as e:
                logger.warning(f"[Server] Failed to parse client parameters: {e}")

        if not weights_results:
            logger.error("[Server] No valid weights received — skipping aggregation.")
            return None, {}

        # Average element-wise across clients
        aggregated_ndarrays = [
            np.mean([weights[i] for weights in weights_results], axis=0)
            for i in range(len(weights_results[0]))
        ]

        # Convert back to Flower Parameters object
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # Log aggregation statistics
        means = [np.mean(arr) for arr in aggregated_ndarrays]
        stds = [np.std(arr) for arr in aggregated_ndarrays]
        logger.info(
            f"[Server] Round {server_round}: aggregated weights "
            f"mean={means}, std={stds}"
        )

        return aggregated_parameters, {}

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


class FedSurvForest(FedAvg):
    """
    Federated Survival Forest (Simple version)

    - Client-proportional sampling
    - Tree-per-client budget correction
    - Uniform tree sampling inside each client
    """

    def __init__(
        self,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        num_trees_fed=None,     # <-- your custom parameter
    ):
        # store custom argument
        self.num_trees_fed = num_trees_fed

        # call FedAvg init with ONLY allowed arguments
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )


    def aggregate_fit(self, server_round, results, failures):
        logger = logging.getLogger("main")

        if not results:
            logger.error("[Server] No results for round %d", server_round)
            return None, {}

        logger.info(f"[Server] Federated RSF aggregation for round {server_round}")

        # ---------------------------------------------------------
        # 1) Extract per-client trees
        # ---------------------------------------------------------
        client_trees = []
        client_sizes = []

        for client_proxy, fit_res in results:
            #trees = fit_res.parameters          # a LIST of sklearn tree estimators
            import pickle
            trees = pickle.loads(fit_res.parameters.tensors[0])
            n_local = fit_res.num_examples      # number of training samples in client
            client_trees.append(trees)
            client_sizes.append(n_local)

            logger.info(
                f"  → Client contributed {len(trees)} trees, "
                f"trained on {n_local} samples"
            )

        client_sizes = np.array(client_sizes, dtype=float)
        total_samples = client_sizes.sum()

        # ---------------------------------------------------------
        # 2) Compute proportional sampling probabilities
        # ---------------------------------------------------------
        client_probs = client_sizes / total_samples
        logger.info(f"[Server] Client sampling probabilities: {client_probs}")

        # ---------------------------------------------------------
        # 3) Total number of federated trees (equals sum locally)
        # ---------------------------------------------------------
        all_local_trees = sum([len(t) for t in client_trees])
        NS = all_local_trees    # simple version: use all local trees

        logger.info(f"[Server] Target federated forest size: {NS} trees")

        # ---------------------------------------------------------
        # 4) Sample client indices proportional to sample counts
        # ---------------------------------------------------------
        sampled_client_ids = np.random.choice(
            a=len(client_trees),
            size=NS,
            p=client_probs
        )

        # Count how many trees we want from each client
        trees_per_client = np.array([
            np.sum(sampled_client_ids == j) for j in range(len(client_trees))
        ])

        logger.info(f"[Server] Desired trees per client: {trees_per_client}")

        # ---------------------------------------------------------
        # 5) Budget correction:
        # If a client does not have enough trees, shift allocation
        # ---------------------------------------------------------
        actual_available = np.array([len(t) for t in client_trees])
        diff = actual_available - trees_per_client   # negative = deficit

        # While ANY client has deficit
        while (diff < 0).any():
            deficit_client = np.random.choice(np.where(diff < 0)[0])
            surplus_client = np.random.choice(np.where(diff > 0)[0])

            trees_per_client[deficit_client] -= 1
            trees_per_client[surplus_client] += 1

            diff = actual_available - trees_per_client

        logger.info(f"[Server] Corrected trees per client: {trees_per_client}")

        # ---------------------------------------------------------
        # 6) Build federated forest (simple version: uniform sampling)
        # ---------------------------------------------------------
        federated_trees = []

        for cid, ntrees in enumerate(trees_per_client):
            local_pool = client_trees[cid]

            if ntrees > 0:
                sampled = np.random.choice(local_pool, size=ntrees, replace=False)
                federated_trees.extend(sampled)

        logger.info(f"[Server] Federated forest built with {len(federated_trees)} trees")

        # ---------------------------------------------------------
        # 7) Return aggregated parameters to clients
        # ---------------------------------------------------------
        import pickle
        from flwr.common import Parameters

        return (
            Parameters(
                tensors=[pickle.dumps(federated_trees)],
                tensor_type="pickle"
            ),
            {}
        )







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
