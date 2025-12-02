from sksurv.ensemble import RandomSurvivalForest
import numpy as np

class SurvivalRandomForest:
    def __init__(self, n_estimators=100, min_samples_split=10, min_samples_leaf=15, random_state=None):
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.global_event_times = None

    # def fit(self, X, y):
    #     self.model.fit(X, y)

    def fit(self, X, y):
        # Apply global time grid BEFORE training the forest
        if self.global_event_times is not None:
            grid = np.asarray(self.global_event_times)
            # Override the RSF internal time grid used by ALL trees
            self.model.unique_times_ = grid
            self.model._n_unique_times = len(grid)
            self.model.event_times_ = grid 
            print(f"[DEBUG][Model] Applied global grid with {len(grid)} points")

        # Train the RSF forest (all trees now consistent)
        self.model.fit(X, y)

        return self


    def predict_survival_function(self, X):
        return self.model.predict_survival_function(X)

    @property
    def estimators_(self):
        return self.model.estimators_

    def set_trees(self, trees, n_features, global_event_times):
        import numpy as np
        from sksurv.util import Surv

        print(f"[DEBUG][set_trees] Received {len(trees)} trees")
        #for i, t in enumerate(trees[:10]):   # show only first 10
            #print(f"    Tree {i}: cumulative_hazard_.shape = {t.cumulative_hazard_.shape}")
        print("    ...")
        print(f"    Desired GLOBAL grid len = {len(global_event_times)}")
        print("-----------------------------------------------------")
        
        # Create dummy dataset to initialize structure
        num_times = len(global_event_times)
        X_dummy = np.zeros((num_times, n_features))
        y_dummy = Surv.from_arrays(
            event=[True] * num_times,
            time=global_event_times,
        )

        # Fit once to initialize RSF internals
        self.model.fit(X_dummy, y_dummy)

        # Override ALL internal time attributes with the global grid
        grid = np.asarray(global_event_times)
        self.model.unique_times_   = grid
        self.model._n_unique_times = len(grid)
        self.model.event_times_    = grid     # <-- ALSO CRITICAL

        # Inject trees
        self.model.estimators_ = trees
        self.model.n_features_in_ = n_features

        print(f"[Client] Loaded federated RSF ({len(trees)} trees) with global grid size {num_times}")
    
    def set_global_time_grid(self, global_times):
        # Store the global grid for use in next fit()
        self.global_event_times = np.array(global_times)
        print(f"[DEBUG][Model] Stored global time grid ({len(self.global_event_times)} points)")

        
    # def set_trees(self, trees, n_features, global_event_times):
    #     import numpy as np
    #     from sksurv.util import Surv

    #     # Create dummy dataset of length == #global time points
    #     num_times = len(global_event_times)
    #     X_dummy = np.zeros((num_times, n_features))
    #     y_dummy = Surv.from_arrays(
    #         event=[True] * num_times,     # dummy event for each time
    #         time=global_event_times
    #     )

    #     # Fit once to initialize event_times_ and internal structures
    #     self.model.fit(X_dummy, y_dummy)

    #     # Inject trees
    #     self.model.estimators_ = trees
    #     self.model.n_features_in_ = n_features
    #     self.model.event_times_ = np.array(global_event_times)

    #     print(f"[Client] Loaded federated RSF ({len(trees)} trees) with global grid size {num_times}")
    


    
    # def set_trees(self, trees):
    #     """
    #     Inject federated trees into a RandomSurvivalForest model.
    #     Ensures internal attributes exist by doing a 1-sample dummy fit.
    #     """

    #     import numpy as np
    #     from sksurv.util import Surv

    #     # ------------------------------------
    #     # 1) Determine number of input features
    #     # ------------------------------------
    #     first_tree = trees[0]

    #     # SurvivalTree stores split features in first_tree.tree_.feature
    #     try:
    #         raw_features = first_tree.tree_.feature
    #         max_feat = raw_features[raw_features >= 0].max()
    #         n_features = int(max_feat + 1)
    #     except Exception:
    #         # fallback: use max_features_
    #         n_features = getattr(first_tree, "max_features_", None)
    #         if n_features is None:
    #             raise ValueError(
    #                 "Unable to infer number of features from federated trees"
    #             )

    #     # ------------------------------------
    #     # 2) If RSF has never been fitted, bootstrap it
    #     # ------------------------------------
    #     if not hasattr(self.model, "event_times_"):
    #         X_dummy = np.zeros((2, n_features)) # RSF needs at least 2 samples to fit
    #         y_dummy = Surv.from_arrays(event=[True, False], time=[1.0, 1.0])

    #         # Fit once to initialize internals
    #         self.model.fit(X_dummy, y_dummy)

    #     # ------------------------------------
    #     # 3) Inject federated trees
    #     # ------------------------------------
    #     self.model.estimators_ = trees

    #     # Debug print
    #     print(f"[Client] Loaded federated forest with {len(trees)} trees")





    # def set_trees(self, trees, n_features):
    #     """
    #     Inject federated trees into a RandomSurvivalForest model.
    #     Ensures internal attributes exist by doing a 1-sample dummy fit.
    #     """

    #     import numpy as np
    #     from sksurv.util import Surv

    #     # Step 1: If RSF has never been fitted, initialize internals
    #     if not hasattr(self.model, "event_times_"):
    #         X_dummy = np.zeros((2, n_features))
    #         y_dummy = Surv.from_arrays(event=[True, False], time=[1.0, 1.0])
    #         self.model.fit(X_dummy, y_dummy)

    #     # Step 2: Inject the federated estimators
    #     self.model.estimators_ = trees

    #     # Step 3: Force correct feature count
    #     self.model.n_features_in_ = n_features

    #     print(f"[Client] Injected federated forest with {len(trees)} trees, features={n_features}")


