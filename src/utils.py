"""Utility functions shared among all training modes."""


import time
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

# Import custom models to check their type
from Models.CustomCoxModel import CustomCoxModel
# Import metrics from lifelines and scikit-survival
from lifelines.utils import concordance_index
from sksurv.metrics import brier_score, cumulative_dynamic_auc
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
from Models.StackedLogisticRegression import StackedLogisticRegression

def train_model(
    model: Any,
    train_data: Any,
    val_data: Any,
    config: Any,
) -> Dict:
    """
    Generic training function that dispatches to the correct
    training logic based on the model type.
    """
    if isinstance(model, torch.nn.Module):
        logging.info("Dispatching to PyTorch model training.")
        return _train_pytorch_model(model, train_data, val_data, config)
    elif isinstance(model, CustomCoxModel):
        logging.info("Dispatching to CustomCoxModel training.")
        return _train_cox_model(model, train_data, val_data, config)
    else:
        raise TypeError(f"Unsupported model type for training: {type(model)}")


def evaluate_model(
    model: Any,
    test_data: Any,
    config: Any,
    train_data: Any = None,
    client_id: int = None,
) -> Dict[str, float]:
    """
    Generic evaluation function for the final test set.
    Dispatches to the correct evaluation logic based on the model type.
    """
    if isinstance(model, torch.nn.Module):
        logging.info("Dispatching to PyTorch model evaluation.")
        return _evaluate_pytorch_model(model, test_data, config.device, config)
    elif isinstance(model, CustomCoxModel):
        logging.info("Dispatching to CustomCoxModel testing.")
        # This function is for the final test evaluation
        return _calculate_cox_metrics(model, test_data, config, train_data=train_data, description="Test")
    elif isinstance(model, CoxPHFitter):
        logging.info("Dispatching to Lifelines CoxPHFitter evaluation.")
        return _evaluate_lifelines_cox(model, test_data, config, train_data=train_data, description="Test", client_id=client_id)
    elif isinstance(model, StackedLogisticRegression):
        logging.info("Dispatching to StackedLogisticRegression evaluation.")
        return _evaluate_stacked_logistic(
            model, test_data, config, train_data=train_data, description="Test", client_id=client_id
        )
    else:
        raise TypeError(f"Unsupported model type for evaluation: {type(model)}")


def _evaluate_lifelines_cox(model, data, config, train_data=None, description="Evaluation", client_id=None):
    """
    Evaluate a fitted lifelines.CoxPHFitter model using:
    - Concordance Index (C-index)
    - Time-dependent AUC (sksurv)
    - Integrated Brier Score (IBS)
    """
    client_tag = f"[Client {client_id}] " if client_id is not None else ""
    logging.info(f"{client_tag}Evaluating Lifelines CoxPH model on {description} data...")

    #logging.info(f"Evaluating Lifelines CoxPH model on {description} data...")

    logging.info("────────────────────────────────────────────────────────")
    logging.info(f"{client_tag} CoxPH Evaluation Results ({description})")
    logging.info("────────────────────────────────────────────────────────")

    # --- 1. Extract features and labels ---
    feature_cols = [c for c in data.columns if c.startswith("feature_")]
    X_test = data[feature_cols]
    T_test = data["time"].values
    E_test = data["event"].astype(int).values

    # --- 2. Concordance Index ---
    try:
        risk_scores = -model.predict_partial_hazard(X_test)
        c_index_test = concordance_index(T_test, risk_scores, E_test)
        logging.info(f"[{description}] C-index: {c_index_test:.6f}")
    except Exception as e:
        logging.warning(f"Failed to compute C-index: {e}")
        c_index_test = np.nan

    # --- 3. Prepare training data (required for time-dependent metrics) ---
    if train_data is None:
        raise ValueError("train_data must be provided to compute AUC and IBS for CoxPH model")

    feature_cols_train = [c for c in train_data.columns if c.startswith("feature_")]
    X_train = train_data[feature_cols_train]
    T_train = train_data["time"].values
    E_train = train_data["event"].astype(int).values

    train_structured = np.array(
        list(zip(E_train.astype(bool), T_train)), dtype=[("event", "bool"), ("time", "f8")]
    )
    test_structured = np.array(
        list(zip(E_test.astype(bool), T_test)), dtype=[("event", "bool"), ("time", "f8")]
    )

    # --- 4. Choose evaluation times ---
    if hasattr(config, "global_eval_times") and config.global_eval_times is not None:
        eval_times = np.array(config.global_eval_times)
        logging.info(f"[{description}] Using global evaluation times from config: {eval_times}")
    else:
        try:
            eval_times = np.quantile(T_test[E_test == 1], np.linspace(0.1, 0.9, 10))
        except Exception:
            eval_times = np.quantile(T_test, np.linspace(0.1, 0.9, 10))
        logging.info(f"[{description}] Using local data-driven eval_times: {eval_times}")

    # Filter times to be within the test data follow-up range
    min_t, max_t = np.min(T_test), np.max(T_test)
    eval_times = eval_times[(eval_times >= min_t) & (eval_times <= max_t)]
    if len(eval_times) == 0:
        logging.warning(f"[{description}] No valid eval_times within [{min_t}, {max_t}] — skipping metrics.")
        return {f"{description.lower()}_c_index": c_index_test, f"{description.lower()}_mean_auc": np.nan, f"{description.lower()}_ibs": np.nan}


    # --- 5. Time-dependent AUC ---
    try:
        aucs, mean_auc = cumulative_dynamic_auc(train_structured, test_structured, risk_scores, eval_times)
        logging.info(f"[{description}] Mean time-dependent AUC: {mean_auc:.6f}")
        logging.info(f"[{description}] Detailed AUC(t):")
        for t, a in zip(eval_times, aucs):
            logging.info(f"   ↳ AUC @ {t:.0f} days = {a:.4f}")
    except Exception as e:
        logging.warning(f"Failed to compute time-dependent AUC: {e}")
        mean_auc = np.nan
        aucs = []


    # --- 6. Integrated Brier Score (IBS) ---
    try:
        # Predict survival probabilities for each test sample at all unique times
        surv_df = model.predict_survival_function(X_test)  # DataFrame: index=time, columns=samples

        # Make sure evaluation times are within the predicted range
        valid_eval_times = [t for t in eval_times if t <= surv_df.index.max()]

        # finds the closest available one (time)
        surv_probs = surv_df.reindex(index=valid_eval_times, method="nearest").fillna(method="ffill").T.values

        # Compute Brier score for each time
        brier_times, brier_scores = brier_score(
            train_structured, test_structured, surv_probs, valid_eval_times
        )
        logging.info(f"[{description}] Detailed Brier(t):")
        for t, b in zip(brier_times, brier_scores):
            logging.info(f"   ↳ Brier @ {t:.0f} days = {b:.4f}")

        # Compute integrated Brier score (area under the curve)
        ibs = np.trapz(brier_scores, brier_times) / (brier_times[-1] - brier_times[0])
        logging.info(f"[{description}] Integrated Brier Score (IBS): {ibs:.6f}")

    except Exception as e:
        logging.warning(f"Failed to compute IBS: {e}")
        ibs = np.nan

    logging.info(f"{client_tag}[{description}] Summary: " 
                 f"C-index={c_index_test:.4f}, Mean AUC={mean_auc:.4f}, IBS={ibs:.4f}")
    logging.info("────────────────────────────────────────────────────────\n")

    return {
        f"{description.lower()}_c_index": float(c_index_test),
        f"{description.lower()}_mean_auc": float(mean_auc),
        f"{description.lower()}_ibs": float(ibs),
    }

# Metrics:
def _calculate_cox_metrics(model, data, config, train_data=None, description="Evaluation"):
    """
    Calculates and returns a dictionary of survival analysis metrics
    using scikit-survival.
    """
    logging.info(f"Calculating Cox metrics for {description} data...")
    
    X = np.array(data["features"])
    T = np.array(data["time"])
    E = np.array(data["event"])

    # --- 1. Concordance Index (from lifelines) ---
    risk_scores = model.predict_risk(X)
    c_index = concordance_index(T, -risk_scores, E)
    logging.info(f"[{description}] Concordance Index: {c_index:.6f}")

    # --- 2. Prepare data for scikit-survival metrics ---
    # If train_data is not provided, try to get it from config (for backward compatibility)
    if train_data is None:
        if hasattr(config, "train_data_for_metrics"):
            train_data = config.train_data_for_metrics
        else:
            raise ValueError(
                "train_data must be provided either directly or via config.train_data_for_metrics."
            )

    train_event_times = np.array(train_data["time"])
    train_event_indicators = np.array(train_data["event"]).astype(bool)
    train_structured_array = np.array(
        list(zip(train_event_indicators, train_event_times)), 
        dtype=[('event', 'bool'), ('time', 'f8')]
    )
    
    test_structured_array = np.array(
        list(zip(E.astype(bool), T)), 
        dtype=[('event', 'bool'), ('time', 'f8')]
    )

    # Define time points for evaluation, ensuring they are within the range of observed times
    min_time = np.min(T[E == 1])
    max_time = np.max(T[E == 1])
    eval_times = np.quantile(T[E == 1], np.linspace(0.1, 0.9, 10))

    # --- 3. Time-Dependent AUC (from sksurv) ---
    try:
        # sksurv's AUC function uses risk scores directly
        auc, mean_auc = cumulative_dynamic_auc(
            train_structured_array, test_structured_array, risk_scores, eval_times
        )
        logging.info(f"[{description}] Mean AUC: {mean_auc:.6f}")
    except Exception as e:
        logging.warning(f"Could not calculate AUC: {e}")
        mean_auc = -1.0 # Placeholder for failure

    # --- 4. Integrated Brier Score (IBS) (from sksurv) ---
    try:
        # sksurv's brier_score needs survival probabilities
        survival_probs_df = model.predict_survival_function(X)
        
        # Ensure the survival function covers the evaluation times
        survival_probs_at_eval_times = survival_probs_df.reindex(eval_times, method='pad').fillna(0)

        # The brier_score function returns scores for each time point.
        # We take the mean to get the Integrated Brier Score.
        brier_scores = brier_score(
            train_structured_array, test_structured_array, survival_probs_at_eval_times.values.T, eval_times
        )
        ibs = np.mean(brier_scores[1])
        logging.info(f"[{description}] Integrated Brier Score: {ibs:.6f}")
    except Exception as e:
        logging.warning(f"Could not calculate IBS: {e}")
        ibs = -1.0 # Placeholder for failure

    return {
        f"{description.lower()}_c_index": c_index,
        f"{description.lower()}_mean_auc": mean_auc,
        f"{description.lower()}_ibs": ibs,
    }


def _train_cox_model(model, train_data, val_data, config):
    """
    Training logic for the CustomCoxModel with per-epoch validation
    and best model checkpointing.
    """
    logging.info("Starting CustomCoxModel training with validation...")
    start_time = time.time()

    X_train = np.array(train_data["features"])
    T_train = np.array(train_data["time"])
    E_train = np.array(train_data["event"])

    best_val_score = -1.0  # We want to maximize C-Index
    best_beta = None
    best_b_j = None
    
    # We need the original training data for weighting in AUC/IBS calculations
    config.train_data_for_metrics = train_data

    for epoch in range(1, config.num_epochs + 1):
        logging.info(f"Epoch [{epoch}/{config.num_epochs}]")
        
        # Train for one epoch
        model.fit(X_train, T_train, E_train, lr=config.lr, epochs=1)

        # Evaluate on the validation set
        val_metrics = _calculate_cox_metrics(model, val_data, config, description="Validation")
        current_val_score = val_metrics.get("validation_c_index", -1.0)

        # Check for improvement and save the best model parameters
        if current_val_score > best_val_score:
            best_val_score = current_val_score
            best_beta = model.get_coefficients().copy()
            best_b_j = model.get_random_effect()
            logging.info(f"New best model found! Validation C-Index: {best_val_score:.6f}")

    # After training, load the best parameters back into the model
    if best_beta is not None:
        logging.info(f"Loading best model with C-Index: {best_val_score:.6f}")
        model.beta = best_beta
        model.b_j = best_b_j
    
    # Finalize the model by computing the baseline hazard with the best parameters
    model.finalize_fit(X_train, T_train, E_train)

    logging.info(f"CustomCoxModel training complete. Total time: {time.time() - start_time:.2f}s")
    
    # Return the metrics from the best model
    final_val_metrics = _calculate_cox_metrics(model, val_data, config, description="Final Validation")
    return final_val_metrics

# Renamed from train_model to be specific to PyTorch
def _train_pytorch_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config,
) -> dict:
    """
    Generic PyTorch training loop for regression, with early stopping.
    """

    model = model.to(config.device)
    # NOTE: This is a regression loss. For survival analysis with PyTorch,
    # you would need a different loss, like a negative partial log-likelihood.
    mse_loss = nn.MSELoss().to(config.device)
    mae_loss = nn.L1Loss().to(config.device)

    # Optimizer
    if config.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.num_epochs + 1):
        logging.info(f"Epoch [{epoch}/{config.num_epochs}]")

        # Training
        model.train()
        train_mse, train_mae = 0.0, 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        start_time = time.time()

        for xb, yb in progress:
            xb, yb = xb.to(config.device), yb.to(config.device)

            optimizer.zero_grad()
            preds = model(xb)
            loss_mse = mse_loss(preds, yb)
            loss_mse.backward()
            optimizer.step()

            train_mse += loss_mse.item()
            train_mae += mae_loss(preds, yb).item()

            progress.set_postfix({"train_mse": f"{loss_mse.item():.4f}"})

        avg_train_mse = train_mse / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)
        logging.info(
            f"Train | MSE={avg_train_mse:.6f}, MAE={avg_train_mae:.6f}, time={time.time() - start_time:.2f}s"
        )

        # Validation
        model.eval()
        val_mse, val_mae = 0.0, 0.0

        progress = tqdm(val_loader, desc=f"Epoch {epoch} [val]")
        start_time = time.time()
        with torch.no_grad():
            for xb, yb in progress:
                xb, yb = xb.to(config.device), yb.to(config.device)
                preds = model(xb)

                vloss = mse_loss(preds, yb)
                val_mse += vloss.item()
                val_mae += mae_loss(preds, yb).item()

                progress.set_postfix({"val_mse": f"{vloss.item():.4f}"})

        avg_val_mse = val_mse / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        logging.info(
            f"Val | MSE={avg_val_mse:.6f}, MAE={avg_val_mae:.6f}, time={time.time() - start_time:.2f}s"
        )

        # Early Stopping 
        if config.early_stopping_patience is not None:
            if best_val_loss - avg_val_mse > config.early_stopping_delta:
                best_val_loss = avg_val_mse
                patience_counter = 0
                # torch.save(model.state_dict(), config.results_path / "best_model.pth")
                logging.info(f"New best model saved (val_mse={best_val_loss:.6f})")
            else:
                patience_counter += 1
                logging.info(
                    f"No improvement ({patience_counter}/{config.early_stopping_patience})"
                )
                if patience_counter >= config.early_stopping_patience:
                    logging.info(f"Early stopping at epoch {epoch}.")
                    break

    # Load best model if available
    # best_model_path = config.results_path / "best_model.pth"
    # if best_model_path.exists():
    #     model.load_state_dict(torch.load(best_model_path, map_location=config.device))
    #     model.to(config.device)

    logging.info("Training complete.")
    return {
        "train_mse": avg_train_mse,
        "train_mae": avg_train_mae,
        "val_mse": avg_val_mse,
        "val_mae": avg_val_mae,
    }



# Renamed from test_model
def _evaluate_pytorch_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config,
) -> Dict[str, float]:
    """
    Generic evaluation function for PyTorch regression models.
    """

    model = model.to(device)
    model.eval()

    mse_criterion = nn.MSELoss(reduction="mean").to(device)
    mae_criterion = nn.L1Loss(reduction="mean").to(device)

    total_mse, total_mae = 0.0, 0.0

    progress = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for xb, yb in progress:
            xb, yb = xb.to(device), yb.to(device)

            preds = model(xb)
            mse = mse_criterion(preds, yb).item()
            mae = mae_criterion(preds, yb).item()

            total_mse += mse
            total_mae += mae

            progress.set_postfix({"mse": f"{mse:.4f}", "mae": f"{mae:.4f}"})

    avg_mse = total_mse / len(test_loader)
    avg_mae = total_mae / len(test_loader)

    logging.info(f"Test Results | MSE={avg_mse:.6f}, MAE={avg_mae:.6f}")

    return {"test_mse": avg_mse, "test_mae": avg_mae}




def _evaluate_stacked_logistic(model, data, config, train_data=None, description="Evaluation", client_id=None):
    """
    Evaluate the Stacked Logistic Regression model using:
      - C-index (based on predicted hazard)
      - AUC (simple binary classification AUC)
      - IBS-like score (approximation)
    """
    client_tag = f"[Client {client_id}] " if client_id is not None else ""
    logging.info(f"{client_tag}Evaluating Stacked Logistic Regression model on {description} data...")
    logging.info("────────────────────────────────────────────────────────")
    logging.info(f"{client_tag} SLR Evaluation Results ({description})")
    logging.info("────────────────────────────────────────────────────────")

    # --- 1. Extract features and labels ---
    # Include both feature_ and time_bin_ columns
    feature_cols = [c for c in data.columns if c.startswith("feature_") or c.startswith("time_bin_")]
    X_test = data[feature_cols]
    y_test = data["event"].astype(int).values

    # --- 2. Predict hazard probabilities ---
    try:
        hazard_pred = model.predict_hazard(X_test)
        logging.info(f"[{description}] Predicted hazard probabilities shape: {hazard_pred.shape}")
    except Exception as e:
        logging.warning(f"Failed to compute hazard predictions: {e}")
        hazard_pred = np.zeros_like(y_test, dtype=float)

    # --- 3. Concordance Index ---
    # Since no survival times are used in SLR directly, we treat hazard_pred as risk score.
    try:
        # pseudo-time ordering if available
        if "time" in data.columns:
            times = data["time"].values
        else:
            times = np.arange(len(y_test))
        c_index_test = concordance_index(times, -hazard_pred, y_test)
        logging.info(f"[{description}] C-index: {c_index_test:.6f}")
    except Exception as e:
        logging.warning(f"Failed to compute C-index: {e}")
        c_index_test = np.nan

    # --- 4. Classic AUC ---
    try:
        auc_test = roc_auc_score(y_test, hazard_pred)
        logging.info(f"[{description}] AUC (classification): {auc_test:.6f}")
    except Exception as e:
        logging.warning(f"Failed to compute AUC: {e}")
        auc_test = np.nan

    # --- 5. IBS-like approximation ---
    # Since we have binary outcomes, use (hazard - event)^2 mean as a proxy
    try:
        ibs_like = np.mean((hazard_pred - y_test) ** 2)
        logging.info(f"[{description}] IBS-like (MSE proxy): {ibs_like:.6f}")
    except Exception as e:
        logging.warning(f"Failed to compute IBS-like metric: {e}")
        ibs_like = np.nan

    logging.info(f"{client_tag}[{description}] Summary: "
                 f"C-index={c_index_test:.4f}, AUC={auc_test:.4f}, IBS-like={ibs_like:.4f}")
    logging.info("────────────────────────────────────────────────────────\n")

    return {
        "C-index": float(c_index_test),
        "AUC": float(auc_test),
        "IBS": float(ibs_like),
    }



import numpy as np
import logging
from lifelines.utils import concordance_index
from sksurv.metrics import (
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    brier_score
)


def evaluate_rsf(model, data, client_id, config):
    """
    Evaluate a Random Survival Forest model using:
        - C-index (Harrell)
        - IPCW C-index (Uno's method)
        - AUC(t) using per-client evaluation times
        - Integrated Brier Score (IBS)

    Parameters
    ----------
    model : SurvivalRandomForest
        The RSF model (with .predict_survival_function and .estimators_)

    data : dict
        {
            "X_test": np.ndarray,
            "y_test": structured array or dataframe,
            "eval_times": np.ndarray,
        }

    client_id : int
    config : Config

    Returns
    -------
    dict with keys: "C-index", "IPCW_C-index", "AUC", "IBS"
    """

    logger = logging.getLogger("main")
    logger.info(f"[Client {client_id}] Evaluating RSF model...")

    # -----------------------------------------------------
    # Unpack data
    # -----------------------------------------------------
    X_test = data["X_test"]
    y_test = data["y_test"]         # structured array
    y_train = data["y_train"]
    eval_times = data["eval_times"]


    # Extract time / event
    test_events = y_test["event"].astype(bool)
    test_times  = y_test["time"].astype(float)

    # -----------------------------------------------------
    # Predict survival curves for test samples
    # -----------------------------------------------------
    surv_fns = model.predict_survival_function(X_test)

    # Convert survival functions into numpy array:
    # shape = (n_samples, n_eval_times)
    surv_probs = np.zeros((len(surv_fns), len(eval_times)))
    for i, fn in enumerate(surv_fns):
        # Evaluate survival fn at all eval_times
        surv_probs[i, :] = fn(eval_times)

    # -----------------------------------------------------
    # C-index (Harrell)
    # -----------------------------------------------------
    try:
        # Use negative survival probability at first eval time as risk score proxy
        risk_scores = -surv_probs[:, 0]  
        c_index = concordance_index(
            test_times,
            risk_scores,
            test_events
        )
    except Exception as e:
        logger.warning(f"[Client {client_id}] Failed to compute C-index: {e}")
        c_index = np.nan

    # -----------------------------------------------------
    # IPCW C-index (Uno method)
    # -----------------------------------------------------
    try:
        cindex_ipcw, _ = concordance_index_ipcw(
            y_train,      # Structured array from training set
            y_test,       # Structured array from test set
            risk_scores
        )
    except Exception as e:
        logger.warning(f"[Client {client_id}] Failed IPCW C-index: {e}")
        cindex_ipcw = np.nan

    # -----------------------------------------------------
    # Time-dependent AUC(t)
    # -----------------------------------------------------
    try:
        aucs, mean_auc = cumulative_dynamic_auc(
            y_train,
            y_test,
            risk_scores,
            eval_times
        )
    except Exception as e:
        logger.warning(f"[Client {client_id}] Failed AUC(t): {e}")
        aucs = []
        mean_auc = np.nan

    # -----------------------------------------------------
    # Integrated Brier Score (IBS)
    # -----------------------------------------------------
    try:
        bs_times, bs_scores = brier_score(
            y_train,
            y_test,
            surv_probs,     # shape = (n_samples, n_times)
            eval_times
        )

        # Numerical integration using trapezoidal rule
        ibs = np.trapz(bs_scores, bs_times) / (bs_times[-1] - bs_times[0])
    except Exception as e:
        logger.warning(f"[Client {client_id}] Failed IBS: {e}")
        ibs = np.nan

    logger.info(
        f"[Client {client_id}] "
        f"C-index={c_index:.4f}, "
        f"IPCW={cindex_ipcw:.4f}, "
        f"AUC={mean_auc:.4f}, "
        f"IBS={ibs:.4f}"
    )

    return {
        "C-index": float(c_index),
        "IPCW_C-index": float(cindex_ipcw),
        "AUC": float(mean_auc),
        "IBS": float(ibs),
        "client_name": f"Client {client_id}"
    }
