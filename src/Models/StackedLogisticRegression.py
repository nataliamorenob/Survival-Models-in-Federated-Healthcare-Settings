import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class StackedLogisticRegression:
    def __init__(self, penalty="l2", C=1.0, random_state=42, solver="liblinear"):
        self.model = LogisticRegression(
            penalty=penalty, C=C, random_state=random_state, solver=solver
        )
        self.scaler = StandardScaler()
        self.time_bins = None

    def fit(self, X, y):
        """
        Fit the stacked logistic regression model.
        Assumes X and y are already transformed.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit logistic regression
        self.model.fit(X_scaled, y)

    def predict_survival_function(self, X, times):
        """
        Predict survival probabilities for given times.
        """
        if self.time_bins is None:
            raise ValueError("Time bins must be set before prediction.")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict hazard for each time bin
        hazard = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate cumulative hazard and survival probability
        survival_probs = []
        for t in times:
            # Find bins up to time t
            relevant_bins = self.time_bins[self.time_bins <= t]
            if len(relevant_bins) == 0:
                survival_probs.append(np.ones(len(X)))
                continue

            # Sum hazards up to time t
            cumulative_hazard = hazard * len(relevant_bins) # Simplified assumption
            survival_prob = np.exp(-cumulative_hazard)
            survival_probs.append(survival_prob)

        return np.array(survival_probs).T

    def get_params(self):
        """
        Return the coefficients of the logistic regression model.
        """
        return [self.model.coef_, self.model.intercept_]

    def set_params(self, params):
        """
        Set the coefficients and intercept of the logistic regression model.
        """
        self.model.coef_ = params[0]
        self.model.intercept_ = params[1]
