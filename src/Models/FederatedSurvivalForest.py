from sksurv.ensemble import RandomSurvivalForest

class SurvivalRandomForest:
    def __init__(self, n_estimators=100, min_samples_split=10, min_samples_leaf=15, random_state=None):
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_survival_function(self, X):
        return self.model.predict_survival_function(X)

    @property
    def estimators_(self):
        return self.model.estimators_

    # def set_trees(self, trees):
    #     """Used by the server to reconstruct the federated forest"""
    #     self.model.estimators_ = trees

    def set_trees(self, trees):
        """
        Inject federated trees into a RandomSurvivalForest model.
        Ensures internal attributes exist by doing a 1-sample dummy fit.
        """

        import numpy as np
        from sksurv.util import Surv

        # ------------------------------------
        # 1) Determine number of input features
        # ------------------------------------
        first_tree = trees[0]

        # SurvivalTree stores split features in first_tree.tree_.feature
        try:
            raw_features = first_tree.tree_.feature
            max_feat = raw_features[raw_features >= 0].max()
            n_features = int(max_feat + 1)
        except Exception:
            # fallback: use max_features_
            n_features = getattr(first_tree, "max_features_", None)
            if n_features is None:
                raise ValueError(
                    "Unable to infer number of features from federated trees"
                )

        # ------------------------------------
        # 2) If RSF has never been fitted, bootstrap it
        # ------------------------------------
        if not hasattr(self.model, "event_times_"):
            X_dummy = np.zeros((2, n_features)) # RSF needs at least 2 samples to fit
            y_dummy = Surv.from_arrays(event=[False, False], time=[1.0, 1.0])

            # Fit once to initialize internals
            self.model.fit(X_dummy, y_dummy)

        # ------------------------------------
        # 3) Inject federated trees
        # ------------------------------------
        self.model.estimators_ = trees

        # Debug print
        print(f"[Client] Loaded federated forest with {len(trees)} trees")