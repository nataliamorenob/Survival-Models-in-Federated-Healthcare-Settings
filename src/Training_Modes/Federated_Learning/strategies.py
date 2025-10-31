# from flwr.server.strategy import FedAvg, FedAdam, FedAdagrad, FedYogi, FedProx

# def get_strategy(strategy_name: str, **kwargs):
#     """
#     Returns the specified federated learning strategy.

#     Parameters:
#         strategy_name (str): The name of the strategy to use (e.g., "FedAvg", "FedAdam").
#         **kwargs: Additional arguments to pass to the strategy.

#     Returns:
#         flwr.server.strategy.FedAvg or other strategy: The selected strategy instance.
#     """
#     strategies = {
#         "FedAvg": FedAvg,
#         "FedAdam": FedAdam,
#         "FedAdagrad": FedAdagrad,
#         "FedYogi": FedYogi,
#         "FedProx": FedProx,
#     }

#     if strategy_name not in strategies:
#         raise ValueError(f"Unsupported strategy: {strategy_name}. Supported strategies are: {list(strategies.keys())}")

#     return strategies[strategy_name](**kwargs)


import numpy as np
import logging
from flwr.server.strategy import FedAvg, FedAdam, FedAdagrad, FedYogi, FedProx


class DebugFedAvg(FedAvg):
    """Custom FedAvg strategy that logs parameter aggregation details."""

    def aggregate_fit(self, server_round, results, failures):
        logger = logging.getLogger("main")
        logger.info(f"\n[Server] ===== Round {server_round} aggregation =====")
        logger.info(f"[Server] Received {len(results)} client results")

        # Inspect one client's parameters to understand the structure
        if results:
            try:
                example_params = results[0][1].parameters.tensors
                logger.info(f"[Server] Example parameter count: {len(example_params)}")

                for i, p in enumerate(example_params[:2]):  # show first 2 arrays
                    arr = np.frombuffer(p, dtype=np.float64)
                    logger.info(
                        f"[Server] Param[{i}] → mean={arr.mean():.6f}, "
                        f"std={arr.std():.6f}, len={len(arr)}"
                    )

            except Exception as e:
                logger.warning(f"[Server] Could not inspect parameters: {e}")

        # Perform standard FedAvg aggregation
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            try:
                agg_tensors = aggregated_parameters.parameters.tensors
                if len(agg_tensors) > 0:
                    arr = np.frombuffer(agg_tensors[0], dtype=np.float64)
                    logger.info(
                        f"[Server] Aggregated Param[0] → mean={arr.mean():.6f}, "
                        f"std={arr.std():.6f}, len={len(arr)}"
                    )
            except Exception as e:
                logger.warning(f"[Server] Could not log aggregated parameters: {e}")

        logger.info(f"[Server] Aggregation done for round {server_round}\n")
        return aggregated_parameters


def get_strategy(strategy_name: str, **kwargs):
    """
    Returns the specified federated learning strategy.
    """
    strategies = {
        "FedAvg": DebugFedAvg,
        "FedAdam": FedAdam,
        "FedAdagrad": FedAdagrad,
        "FedYogi": FedYogi,
        "FedProx": FedProx,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unsupported strategy: {strategy_name}. Supported: {list(strategies.keys())}"
        )

    logger = logging.getLogger("main")
    logger.info(f"[Strategy] Selected strategy: {strategy_name}")

    return strategies[strategy_name](**kwargs)
