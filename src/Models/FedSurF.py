# Models/FederatedSurvivalForest.py
from sksurv.ensemble import RandomSurvivalForest
import numpy as np
from scipy.interpolate import interp1d


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

    def predict_survival_function_fedsurf(self, X):
        """
        FedSurF-style prediction:
          - predict survival via individual trees
          - collect all time grids
          - build union grid
          - interpolate each tree fn
          - average them
        """

        trees = self.model.estimators_
        n_trees = len(trees)
        n_samples = X.shape[0]

        # ----------------------------------------------------
        # 1) Get tree-level predictions (each tree has its own time grid)
        # ----------------------------------------------------
        tree_survs = []
        tree_times = []

        for t in trees:
            surv_fns = t.predict_survival_function(X)  # list length n_samples
            tree_survs.append(surv_fns)
            tree_times.append(t.unique_times_)  # each tree has its own grid
        # tree_survs[j][i] = survival curve from tree j for sample i
        # tree_times[j] = time grid of tree j


        # ----------------------------------------------------
        # 2) Build the global union of all time grids
        # ----------------------------------------------------
        global_times = np.unique(np.concatenate(tree_times))

        # ----------------------------------------------------
        # 3) Interpolate each tree's survival curve to global grid
        # ----------------------------------------------------
        out_surv = []

        for i in range(n_samples):

            # store all interpolated curves from all trees
            curves = []

            for j in range(n_trees):
                fn = tree_survs[j][i]       # survival_function object
                t_local = tree_times[j]
                s_local = fn(t_local)

                # monotonic interpolation
                f = interp1d(
                    t_local, s_local, kind="previous", bounds_error=False,
                    fill_value=(1.0, s_local[-1])
                )

                curves.append(f(global_times))

            # average survival curve over trees
            mean_curve = np.mean(curves, axis=0)
            out_surv.append((global_times, mean_curve))

        return out_surv


    # def set_trees(self, trees, n_features):
    #     """
    #     Load a federated global forest made of trees from many clients.
    #     """
    #     self.model.estimators_ = trees
    #     self.model.n_features_in_ = n_features
