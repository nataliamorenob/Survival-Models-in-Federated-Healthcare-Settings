import numpy as np
import pandas as pd

class CustomCoxModel:
    def __init__(self):
        self.beta = None # Global coefficients (fixed effects)
        self.b_j = 0.0 # Local random effect
        self.baseline_cumulative_hazard_ = None
        self.event_times_ = None

    def _run_training_epoch(self, X, T, E, lr):
        """
        Runs a single epoch of gradient descent.
        """
        n_samples, n_features = X.shape
        
        risk_scores = np.dot(X, self.beta) + self.b_j # (Xiβ+bj)
        exp_risk_scores = np.exp(risk_scores)

        partial_likelihood_beta = np.zeros(n_features)
        partial_likelihood_bj = 0.0

        risk_set_sums = np.array([np.sum(exp_risk_scores[i:]) for i in range(n_samples)])

        for i in range(n_samples):
            if E[i] == 1:
                risk_set_sum = risk_set_sums[i]
                if risk_set_sum > 0:
                    weighted_covariates_sum = np.sum(X[i:] * (exp_risk_scores[i:] / risk_set_sum)[:, None], axis=0)
                    partial_likelihood_beta += X[i] - weighted_covariates_sum
                    partial_likelihood_bj += 1 - np.sum(exp_risk_scores[i:] / risk_set_sum)

        self.beta += lr * partial_likelihood_beta
        self.b_j += lr * partial_likelihood_bj

    def fit(self, X, T, E, lr=0.01, epochs=100):
        """
        This is now a simple wrapper for a single epoch for use in the main training loop.
        The baseline hazard is computed separately.
        """
        n_samples, n_features = X.shape
        if self.beta is None:
            self.beta = np.zeros(n_features)

        # Sort data by time for correct risk set calculation
        order = np.argsort(T)
        X_sorted, T_sorted, E_sorted = X[order], T[order], E[order]
        
        self._run_training_epoch(X_sorted, T_sorted, E_sorted, lr)

    def finalize_fit(self, X, T, E):
        """
        Computes the baseline cumulative hazard function using the Breslow estimator.
        This should be called once after the best model parameters are determined.
        """
        # Sort data just in case
        order = np.argsort(T)
        X_sorted, T_sorted, E_sorted = X[order], T[order], E[order]

        base_risk_scores = np.exp(np.dot(X_sorted, self.beta))
        
        unique_event_times = np.unique(T_sorted[E_sorted == 1])
        
        baseline_hazard = []
        for t in unique_event_times:
            risk_set_sum = np.sum(base_risk_scores[T_sorted >= t])
            events_at_t = np.sum(E_sorted[T_sorted == t])
            
            if risk_set_sum > 0:
                baseline_hazard.append(events_at_t / risk_set_sum)
        
        self.baseline_cumulative_hazard_ = np.cumsum(baseline_hazard)
        self.event_times_ = unique_event_times

    def predict_survival_function(self, X_new):
        """
        Predict the survival function for new individuals.
        """
        if self.baseline_cumulative_hazard_ is None:
            raise RuntimeError("The model has not been finalized. Call `finalize_fit` first.")

        risk_scores = np.exp(np.dot(X_new, self.beta) + self.b_j)
        individual_cumulative_hazards = np.outer(self.baseline_cumulative_hazard_, risk_scores)
        survival_probabilities = np.exp(-individual_cumulative_hazards)
        
        return pd.DataFrame(survival_probabilities, index=self.event_times_, columns=range(X_new.shape[0]))

    def predict_risk(self, X):
        """ Predict risk scores for new data. """
        return np.dot(X, self.beta) + self.b_j

    def get_coefficients(self):
        """Return model coefficients (fixed effects) as a list of NumPy arrays."""
        if self.beta is None:
            # initialize to random or zeros — both are okay
            self.beta = np.zeros(39)  
        return [self.beta]  # Flower expects a list of NumPy arrays


    def get_random_effect(self):
        """ Get the local random effect. """
        return self.b_j