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
