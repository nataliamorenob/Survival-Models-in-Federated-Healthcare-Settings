# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler

# class StackedLogisticRegression:
#     def __init__(self, penalty="l2", C=1.0, random_state=42, solver="liblinear"):
#         self.model = LogisticRegression(
#             penalty=penalty, C=C, random_state=random_state, solver=solver
#         )
#         self.scaler = StandardScaler()
#         self.time_bins = None

#     def fit(self, X, y):
#         """
#         Fit the stacked logistic regression model.
#         Assumes X and y are already transformed.
#         """
#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)

#         # Fit logistic regression
#         self.model.fit(X_scaled, y)

#     def predict_survival_function(self, X, times):
#         """
#         Predict survival probabilities for given times.
#         """
#         if self.time_bins is None:
#             raise ValueError("Time bins must be set before prediction.")

#         # Scale features
#         X_scaled = self.scaler.transform(X)

#         # Predict hazard for each time bin
#         hazard = self.model.predict_proba(X_scaled)[:, 1]
        
#         # Calculate cumulative hazard and survival probability
#         survival_probs = []
#         for t in times:
#             # Find bins up to time t
#             relevant_bins = self.time_bins[self.time_bins <= t]
#             if len(relevant_bins) == 0:
#                 survival_probs.append(np.ones(len(X)))
#                 continue

#             # Sum hazards up to time t
#             cumulative_hazard = hazard * len(relevant_bins) # Simplified assumption
#             survival_prob = np.exp(-cumulative_hazard)
#             survival_probs.append(survival_prob)

#         return np.array(survival_probs).T

#     def get_params(self):
#         """
#         Return the coefficients of the logistic regression model.
#         """
#         return [self.model.coef_, self.model.intercept_]

#     def set_params(self, params):
#         """
#         Set the coefficients and intercept of the logistic regression model.
#         """
#         self.model.coef_ = params[0]
#         self.model.intercept_ = params[1]





# MODEL DOES NOT UPDATE THE WEIGHTS I THINK BC HOW LOGISTICREGRESSION WORKS INSIDE
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler

# class StackedLogisticRegression:
#     def __init__(
#         self,
#         penalty="l2",
#         C=1.0,
#         random_state=42,
#         solver="liblinear",
#         max_iter=100,
#     ):
#         # Store hyperparameters
#         self.penalty = penalty
#         self.C = C
#         self.random_state = random_state
#         self.solver = solver
#         self.max_iter = max_iter

#         self.model = LogisticRegression(
#             penalty=self.penalty,
#             C=self.C,
#             random_state=self.random_state,
#             solver=self.solver,
#             max_iter=self.max_iter,
#             warm_start=True,
#         )
#         self.fitted = False

#     def fit(self, X, y):
#         """Fit the logistic regression model (already stacked and binary)."""
#         self.model.fit(X, y)
#         self.fitted = True

#     def predict_hazard(self, X):
#         """Predict hazard probabilities for each observation."""
#         import pandas as pd

#         # Defensive check to ensure same feature count as during training
#         if isinstance(X, pd.DataFrame):
#             # Drop any label/time columns that shouldn't be there
#             X = X.drop(columns=["event", "time"], errors="ignore")

#         # If feature mismatch persists, auto-truncate to expected number
#         if hasattr(self.model, "n_features_in_") and X.shape[1] != self.model.n_features_in_:
#             X = X.iloc[:, : self.model.n_features_in_]

#         preds = self.model.predict_proba(X)[:, 1]
#         return preds


#     def get_params(self):
#         """Return model parameters if fitted, otherwise initialize them."""
#         import numpy as np

#         if hasattr(self.model, "coef_") and hasattr(self.model, "intercept_"):
#             return [self.model.coef_, self.model.intercept_]
#         else:
#             # Model not yet fitted — initialize empty params based on input size if available
#             n_features = getattr(self, "n_features_in_", None)
#             if n_features is not None:
#                 coef = np.zeros((1, n_features))
#                 intercept = np.zeros(1)
#             else:
#                 # fallback for very first round
#                 coef = np.zeros((1, 1))
#                 intercept = np.zeros(1)
#             return [coef, intercept]


#     def set_params(self, params):
#         """
#         Update model weights (without reinitializing LogisticRegression).
#         Ensures the model continues training from server weights.
#         """
#         import numpy as np

#         # Ensure model exists
#         if not hasattr(self, "model") or self.model is None:
#             from sklearn.linear_model import LogisticRegression
#             self.model = LogisticRegression(
#                 penalty=self.penalty,
#                 C=self.C,
#                 solver=self.solver,
#                 max_iter=self.max_iter,
#                 warm_start=True,
#                 random_state=self.random_state,
#             )

#         # Update existing model weights
#         self.model.classes_ = np.array([0, 1])
#         self.model.coef_ = params[0].copy()
#         self.model.intercept_ = params[1].copy()
#         self.model.n_features_in_ = self.model.coef_.shape[1]


#     def set_weights(self, params):
#         self.set_params(params)




from sklearn.linear_model import SGDClassifier
import numpy as np

class StackedLogisticRegression:
    def __init__(
        self,
        penalty="l2",
        C=1.0,
        random_state=42,
        max_iter=100,
        learning_rate="optimal",
        eta0=0.01,
    ):
        self.penalty = penalty
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.eta0 = eta0

        # SGDClassifier with log_loss behaves like logistic regression but supports partial_fit
        # linear model trained using stochastic gradient descent (SGD)
        self.model = SGDClassifier(
            loss="log_loss",
            penalty=self.penalty,
            alpha=1.0 / self.C,  # inverse of C
            random_state=self.random_state,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            eta0=0.1,  # Increased learning rate
            warm_start=True,  # Enable warm_start
        )
        self.fitted = False

    def fit(self, X, y):
        import numpy as np
        import pandas as pd

        # --- make sure every feature is float32 ---
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        # ------------------------------------------

        if not self.fitted:
            self.model.partial_fit(X, y, classes=np.array([0, 1]))
            self.fitted = True
        else:
            self.model.partial_fit(X, y)


    def predict_hazard(self, X):
        """Predict hazard probabilities for each observation."""
        import pandas as pd
        import numpy as np

        # Defensive check: drop non-feature columns if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.drop(columns=["event", "time"], errors="ignore")
            X = X.to_numpy(dtype=np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        # If feature mismatch persists, adjust size safely
        if hasattr(self.model, "n_features_in_") and X.shape[1] != self.model.n_features_in_:
            X = X[:, : self.model.n_features_in_]  # use numpy slicing (not .iloc)

        # Compute probabilities or fallback
        if hasattr(self.model, "predict_proba"):
            preds = self.model.predict_proba(X)[:, 1]
        else:
            preds = 1 / (1 + np.exp(-self.model.decision_function(X)))

        return preds



    def get_params(self):
        """Return model parameters, initializing if model not yet trained."""
        import numpy as np

        # If model is already fitted, return weights
        if hasattr(self.model, "coef_") and hasattr(self.model, "intercept_"):
            return [self.model.coef_, self.model.intercept_]

        # If model not yet trained, initialize to correct feature size (39)
        n_features = 139  # fixed number of features in stacked dataset
        coef = np.zeros((1, n_features), dtype=np.float32)
        intercept = np.zeros(1, dtype=np.float32)

        return [coef, intercept]



    def set_params(self, params):
        """Set model parameters from a list of NumPy arrays."""
        self.model.coef_ = params[0].copy()
        self.model.intercept_ = params[1].copy()
        self.model.classes_ = np.array([0, 1])
        
        # Explicitly update the number of features the model expects
        self.model.n_features_in_ = self.model.coef_.shape[1]
        
        self.fitted = True

