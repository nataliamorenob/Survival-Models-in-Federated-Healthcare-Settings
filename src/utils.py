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
        return _calculate_cox_metrics(model, test_data, config, description="Test")
    else:
        raise TypeError(f"Unsupported model type for evaluation: {type(model)}")




# Metrics:
def _calculate_cox_metrics(model, data, config, description="Evaluation"):
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
    # sksurv expects structured arrays for survival data.
    train_event_times = np.array(config.train_data_for_metrics["time"])
    train_event_indicators = np.array(config.train_data_for_metrics["event"]).astype(bool)
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
