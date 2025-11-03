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
            client_name = getattr(client_proxy, "name", f"Client {client_id}")
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



def get_strategy(strategy_name: str, **kwargs):
    """Return the selected FL strategy (CustomFedAvg by default)."""
    strategies = {
        "FedAvg": CustomFedAvg,
        "FedAdam": FedAdam,
        "FedAdagrad": FedAdagrad,
        "FedYogi": FedYogi,
        "FedProx": FedProx,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unsupported strategy: {strategy_name}. "
            f"Supported: {list(strategies.keys())}"
        )

    logger = logging.getLogger("main")
    logger.info(f"[Strategy] Selected strategy: {strategy_name}")

    return strategies[strategy_name](**kwargs)
