"""Cox Proportional Hazards Model: How much more (or less) likely is a certain outcome 
in one group compared to another, at any given time?"""

from lifelines import CoxPHFitter
import pickle
import os

def CoxPH_model(df, config, client_id, duration_col='time', event_col='event'):
    """
    Fits a Cox Proportional Hazards model to the given DataFrame.

    Parameters:
        df (pd.DataFrame): The preprocessed DataFrame containing the data.
        config (Config): The experiment configuration object.
        client_id (int): The ID of the client.
        duration_col (str): The name of the column representing time duration.
        event_col (str): The name of the column representing the event indicator (1 for event, 0 for censoring).

    Returns:
        None
    """
    # Initialize the Cox Proportional Hazards model
    cph = CoxPHFitter(penalizer=1.0)

    # Fit the model to the data
    cph.fit(df, duration_col=duration_col, event_col=event_col)

    # Print the summary of the model
    cph.print_summary()

    # Visualize the coefficients
    cph.plot()

    # Save safely with pickle
    model_path = os.path.join(config.experiment_dir, f"cox_model_client_{client_id}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(cph, f)

    return cph
