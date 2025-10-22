import flwr as fl
import numpy as np

class FederatedCoxServer(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        """
        Aggregate the coefficients from clients.

        Parameters:
            rnd (int): Current round number.
            results (List[Tuple[fl.common.Parameters, int]]): List of parameters and sample sizes.
            failures (List[BaseException]): List of exceptions raised by clients.

        Returns:
            fl.common.Parameters: Aggregated parameters.
        """
        if failures:
            print(f"Failures during round {rnd}: {failures}")

        # Extract coefficients and sample sizes
        coefficients = [np.array(fit_res.parameters) for fit_res, _ in results]
        sample_sizes = [num_examples for _, num_examples in results]

        # Weighted average of coefficients
        total_samples = sum(sample_sizes)
        aggregated_coefficients = np.sum(
            [coeff * (size / total_samples) for coeff, size in zip(coefficients, sample_sizes)], axis=0
        )

        return aggregated_coefficients

# Example usage
# strategy = FederatedCoxServer()
# fl.server.start_server("localhost:8080", config={"num_rounds": 10}, strategy=strategy)