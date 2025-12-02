# Models/FederatedSurvivalForest.py
from sksurv.ensemble import RandomSurvivalForest
import numpy as np

class SurvivalRandomForest:
    """
    Clean RSF wrapper for FedSurF:
      - fit() trains a local forest
      - estimators_ exposes the trained trees
      - set_trees() loads a global federated forest
    """

    def __init__(self, n_estimators=100, min_samples_split=10, min_samples_leaf=15, random_state=None):
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(self, X, y):
        """Train local RSF."""
        self.model.fit(X, y)
        return self

    @property
    def estimators_(self):
        """Return trained local trees."""
        return self.model.estimators_

    def set_trees(self, trees, n_features):
        """
        Load a federated global forest made of trees from many clients.
        """
        self.model.estimators_ = trees
        self.model.n_features_in_ = n_features

    def predict_survival_function(self, X):
        return self.model.predict_survival_function(X)