from flwr.server.strategy import FedAvg, FedAdam, FedAdagrad, FedYogi, FedProx

def get_strategy(strategy_name: str, **kwargs):
    """
    Returns the specified federated learning strategy.

    Parameters:
        strategy_name (str): The name of the strategy to use (e.g., "FedAvg", "FedAdam").
        **kwargs: Additional arguments to pass to the strategy.

    Returns:
        flwr.server.strategy.FedAvg or other strategy: The selected strategy instance.
    """
    strategies = {
        "FedAvg": FedAvg,
        "FedAdam": FedAdam,
        "FedAdagrad": FedAdagrad,
        "FedYogi": FedYogi,
        "FedProx": FedProx,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unsupported strategy: {strategy_name}. Supported strategies are: {list(strategies.keys())}")

    return strategies[strategy_name](**kwargs)