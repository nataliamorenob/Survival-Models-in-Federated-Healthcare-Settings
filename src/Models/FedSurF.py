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


    def predict_survival_function(self, X):
        return self.model.predict_survival_function(X)

    def set_trees(self, trees, n_features):
        """
        Load a federated global forest made of trees from many clients.
        A dummy fit is required to initialize RSF internal attributes.
        """
        import numpy as np
        from sksurv.util import Surv

        # 1) Dummy tiny dataset to initialize RSF internals:
        # one sample, one fake time, event=True
        X_dummy = np.zeros((2, n_features))
        y_dummy = Surv.from_arrays(event=[True, True], time=[1.0, 2.0])

        # this creates n_outputs_, unique_times_, event_times_, etc.
        self.model.fit(X_dummy, y_dummy)

        # 2) Inject trees AFTER initialization:
        self.model.estimators_ = trees
        self.model.n_features_in_ = n_features


    # def set_trees(self, trees, n_features):
    #     """
    #     Load a federated global forest made of trees from many clients.
    #     """
    #     self.model.estimators_ = trees
    #     self.model.n_features_in_ = n_features
