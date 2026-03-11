# Models/RSF_FedSurF.py
"""
FedSurF++ Model Implementation
Random Survival Forest wrapper for FedSurF++ with C-Index based tree sampling.
"""

from sksurv.ensemble import RandomSurvivalForest
import numpy as np
from scipy.interpolate import interp1d


class RSFFedSurFPlus:
    """
    Random Survival Forest wrapper for FedSurF++:
      - fit() trains a local forest
      - estimators_ exposes the trained trees
      - set_trees() loads a global federated forest
      - predict_survival_function_fedsurf() averages tree predictions
    """

    def __init__(self, n_estimators=100, min_samples_split=10, min_samples_leaf=15, random_state=None):
        """
        Initialize RSF model for FedSurF++.
        
        Args:
            n_estimators: Number of trees in local forest (n_trees_local)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            random_state: Random seed for reproducibility
        """
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(self, X, y):
        """Train local RSF on client data."""
        self.model.fit(X, y)
        return self

    @property
    def estimators_(self):
        """Return trained local trees."""
        return self.model.estimators_

    def predict_survival_function(self, X):
        """Standard RSF prediction."""
        return self.model.predict_survival_function(X)

    def set_trees(self, trees, n_features, init_times):
        """
        Load global federated forest from server.
        
        Args:
            trees: List of decision trees from different clients
            n_features: Number of input features
            init_times: Union time grid for proper initialization
        """
        from sksurv.util import Surv

        if init_times is None or len(init_times) < 3:
            raise ValueError(
                "set_trees() requires init_times (union_time_grid) with real survival times."
            )

        # 1. Initialize RSF with dummy data but real survival times
        X_init = np.zeros((len(init_times), n_features))
        y_init = Surv.from_arrays(
            event=np.ones(len(init_times), dtype=bool),
            time=init_times,
        )

        # 2. Fit once to initialize internal RSF hazard structure
        self.model.fit(X_init, y_init)

        # 3. Inject federated trees
        self.model.estimators_ = trees
        self.model.n_features_in_ = n_features

    def predict_survival_function_fedsurf(self, X):
        """
        FedSurF-style prediction with tree averaging.
        
        Steps:
        1. Get predictions from each tree (each has its own time grid)
        2. Build union of all time grids
        3. Interpolate each tree's survival curve to union grid
        4. Average survival curves across all trees
        
        Returns:
            List of (times, survival_probs) tuples for each sample
        """
        trees = self.model.estimators_
        n_trees = len(trees)
        n_samples = X.shape[0]

        # 1) Get tree-level predictions
        tree_survs = []
        tree_times = []

        for t in trees:
            surv_fns = t.predict_survival_function(X)
            tree_survs.append(surv_fns)
            tree_times.append(t.unique_times_)

        # 2) Build global union of all time grids
        global_times = np.unique(np.concatenate(tree_times))

        # 3) Interpolate and average
        out_surv = []

        for i in range(n_samples):
            curves = []

            for j in range(n_trees):
                fn = tree_survs[j][i]
                t_local = tree_times[j]
                s_local = fn(t_local)

                # Monotonic interpolation
                f = interp1d(
                    t_local, s_local, kind="previous", bounds_error=False,
                    fill_value=(1.0, s_local[-1])
                )

                curves.append(f(global_times))

            # Average survival curve over trees
            mean_curve = np.mean(curves, axis=0)
            out_surv.append((global_times, mean_curve))

        return out_surv
