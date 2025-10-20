"""Cox Proportional Hazards Model: How much more (or less) likely is a certain outcome 
in one group compared to another, at any given time?"""

from lifelines import CoxPHFitter

def CoxPH_model(df, duration_col='time', event_col='event', model_path='cox_model.pkl'):
    """
    Fits a Cox Proportional Hazards model to the given DataFrame.

    Parameters:
        df (pd.DataFrame): The preprocessed DataFrame containing the data.
        duration_col (str): The name of the column representing time duration.
        event_col (str): The name of the column representing the event indicator (1 for event, 0 for censoring).
        model_path (str): The path to save the fitted model.

    Returns:
        None
    """
    # Initialize the Cox Proportional Hazards model
    cph = CoxPHFitter()

    # Fit the model to the data
    cph.fit(df, duration_col=duration_col, event_col=event_col)

    # Print the summary of the model
    cph.print_summary()

    # Visualize the coefficients
    cph.plot()

    # Save the model for later use
    cph.save_model(model_path)
