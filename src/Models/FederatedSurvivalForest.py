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

    def set_trees(self, trees):
        """Used by the server to reconstruct the federated forest"""
        self.model.estimators_ = trees