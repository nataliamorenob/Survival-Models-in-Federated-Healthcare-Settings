import numpy as np
import pandas as pd

class CustomCoxModel:
    def __init__(self):
        self.beta = None  # Global coefficients (fixed effects)
        self.b_j = 0.0  # Local random effect

    def fit(self, X, T, E, lr=0.01, epochs=100):
        """
        Fit the mixed Cox model using partial likelihood.

        Parameters:
            X (np.ndarray): Feature matrix (n_samples, n_features).
            T (np.ndarray): Time-to-event data.
            E (np.ndarray): Event indicator (1 if event occurred, 0 if censored).
            lr (float): Learning rate for gradient descent.
            epochs (int): Number of training epochs.
        """
        n_samples, n_features = X.shape
        self.beta = np.zeros(n_features)  # Initialize fixed effects

        # Sort by time (ascending)
        order = np.argsort(T)
        X, T, E = X[order], T[order], E[order]


        for epoch in range(epochs):
            risk_scores = np.dot(X, self.beta) + self.b_j
            exp_risk_scores = np.exp(risk_scores)

            # Compute the partial likelihood gradient for beta and b_j
            partial_likelihood_beta = np.zeros(n_features)
            partial_likelihood_bj = 0.0

            for i in range(n_samples):
                if E[i] == 1:  # Only consider events
                    risk_set = exp_risk_scores[i:]
                    partial_likelihood_beta += X[i] - np.sum(X[i:] * (risk_set / np.sum(risk_set))[:, None], axis=0)
                    partial_likelihood_bj += 1 - np.sum(risk_set / np.sum(risk_set))

            # Update beta and b_j using gradient descent
            self.beta += lr * partial_likelihood_beta
            self.b_j += lr * partial_likelihood_bj

    def predict_risk(self, X):
        """
        Predict risk scores for new data.

        Parameters:
            X (np.ndarray): Feature matrix (n_samples, n_features).

        Returns:
            np.ndarray: Risk scores.
        """
        return np.dot(X, self.beta) + self.b_j

    def get_coefficients(self): 
        """ Get the model coefficients (fixed effects). Returns: np.ndarray: Coefficients (beta). """ 
        return self.beta

    def get_random_effect(self):
        """
        Get the local random effect.

        Returns:
            float: Random effect (b_j).
        """
        return self.b_j