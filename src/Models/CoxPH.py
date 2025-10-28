"""Cox Proportional Hazards Model: How much more (or less) likely is a certain outcome 
in one group compared to another, at any given time?"""

from lifelines import CoxPHFitter
import pickle
import os

import pandas as pd
from config import ALL_FEATURE_COLUMNS
from lifelines import CoxPHFitter
import pickle
import os

# def CoxPH_model(df, config, client_id, duration_col='time', event_col='event', init_params=None, max_iter=100):
#     """
#     Fits a Cox Proportional Hazards model to the given DataFrame.

#     Parameters:
#         df (pd.DataFrame): The preprocessed DataFrame containing the data.
#         config (Config): The experiment configuration object.
#         client_id (int): The ID of the client.
#         duration_col (str): The name of the column representing time duration.
#         event_col (str): The name of the column representing the event indicator (1 for event, 0 for censoring).
#         init_params (np.ndarray, optional): Initial parameters for the model. Defaults to None.
#         max_iter (int): The maximum number of iterations for the fitter.

#     Returns:
#         None
#     """
#     # Initialize the Cox Proportional Hazards model
#     cph = CoxPHFitter(penalizer=1.0)

#     initial_point = None
#     if init_params is not None:
#         # Create a Series with all feature columns, then select only those present in df
#         initial_params_series = pd.Series(init_params, index=ALL_FEATURE_COLUMNS)
#         # Align with the dataframe's columns, excluding duration and event columns
#         feature_cols = [col for col in df.columns if col not in [duration_col, event_col]]
#         initial_point = initial_params_series.reindex(feature_cols).fillna(0).values

#     # Fit the model to the data
#     cph.fit(df, duration_col=duration_col, event_col=event_col, initial_point=initial_point, fit_options={'max_steps': max_iter})

#     # Print the summary of the model
#     cph.print_summary()

#     # Visualize the coefficients
#     cph.plot()

#     # Save safely with pickle
#     model_path = os.path.join(config.experiment_dir, f"cox_model_client_{client_id}.pkl")
#     with open(model_path, "wb") as f:
#         pickle.dump(cph, f)

#     return cph


# def CoxPH_model(df, config, client_id, duration_col='time', event_col='event', init_params=None, max_iter=100):
#     """
#     Fits a Cox Proportional Hazards model, optionally initializing with given parameters.
#     """
#     from lifelines import CoxPHFitter
#     import pandas as pd
#     import numpy as np
#     import os
#     import pickle
#     from config import ALL_FEATURE_COLUMNS

#     cph = CoxPHFitter(penalizer=1.0)

#     # Fit from scratch
#     cph.fit(df, duration_col=duration_col, event_col=event_col, fit_options={'max_steps': max_iter})

#     # --- Inject global parameters (warm start from global model) ---
#     if init_params is not None:
#         # Align with features in df
#         initial_params_series = pd.Series(init_params, index=ALL_FEATURE_COLUMNS)
#         feature_cols = [col for col in df.columns if col not in [duration_col, event_col]]
#         global_beta = initial_params_series.reindex(feature_cols).fillna(0)

#         # Override the fitted coefficients
#         cph.params_ = global_beta
#         cph._compute_baseline_hazard(df, duration_col, event_col)
#         cph._compute_baseline_cumulative_hazard()
#         cph._compute_baseline_survival_function()

#     # Save model
#     model_path = os.path.join(config.experiment_dir, f"cox_model_client_{client_id}.pkl")
#     with open(model_path, "wb") as f:
#         pickle.dump(cph, f)

#     return cph


def CoxPH_model(df, config, client_id, duration_col='time', event_col='event', init_params=None, max_iter=100):
    """
    Fits a Cox Proportional Hazards model, optionally initializing with given parameters.
    """
    from lifelines import CoxPHFitter
    import pandas as pd
    import numpy as np
    import os
    import pickle
    from config import ALL_FEATURE_COLUMNS

    cph = CoxPHFitter(penalizer=1.0)

    # Prepare initialization point if global parameters are given
    initial_point = None
    if init_params is not None:
        initial_params_series = pd.Series(init_params, index=ALL_FEATURE_COLUMNS)
        feature_cols = [col for col in df.columns if col not in [duration_col, event_col]]
        # Align to features present in this client's data
        initial_point = initial_params_series.reindex(feature_cols).fillna(0).values

    # Fit the model (using global β as a warm start if provided)
    cph.fit(
        df,
        duration_col=duration_col,
        event_col=event_col,
        initial_point=initial_point,   # Lifelines supports this officially
        fit_options={"max_steps": max_iter}
    )

    # Save for debugging / later reuse
    model_path = os.path.join(config.experiment_dir, f"cox_model_client_{client_id}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(cph, f)

    return cph
