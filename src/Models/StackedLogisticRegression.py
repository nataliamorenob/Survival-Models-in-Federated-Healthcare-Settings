import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class StackedLogisticRegression:
    def __init__(self, penalty="l2", C=1.0, random_state=42, solver="liblinear"):
        self.model = LogisticRegression(
            penalty=penalty, C=C, random_state=random_state, solver=solver
        )
        self.scaler = StandardScaler()
        self.time_bins = None

    def _transform_data(self, X, y, time_bins):
        """
        Transforms survival data into a classification-style dataset.
        For each patient and each time bin, we create a row.
        The target is 1 if the patient fails in that bin, 0 otherwise.
        """
        X_transformed, y_transformed = [], []
        
        for i in range(len(X)):
            for t_bin in time_bins:
                # Add features
                X_transformed.append(X.iloc[i].values)
                
                # Create target: 1 if event happened in this bin, 0 otherwise
                event_in_bin = 1 if y["time"].iloc[i] <= t_bin and y["event"].iloc[i] == 1 else 0
                y_transformed.append(event_in_bin)

        return pd.DataFrame(X_transformed, columns=X.columns), pd.Series(y_transformed)

    def fit(self, X, y):
        """
        Fit the stacked logistic regression model.
        """
        # Discretize time into bins
        self.time_bins = np.quantile(y[y["event"] == 1]["time"], q=np.linspace(0.1, 1, 10)).unique()

        # Transform data
        X_train, y_train = self._transform_data(X, y, self.time_bins)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Fit logistic regression
        self.model.fit(X_train_scaled, y_train)

    def predict_survival_function(self, X, times):
        """
        Predict survival probabilities for given times.
        """
        if self.time_bins is None:
            raise ValueError("The model must be fitted before prediction.")

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
        return self.model.coef_

    def set_params(self, params):
        """
        Set the coefficients of the logistic regression model.
        """
        self.model.coef_ = params
