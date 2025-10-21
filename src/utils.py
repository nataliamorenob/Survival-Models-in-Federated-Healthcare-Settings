"""Utility functions shared among all training modes."""


import time
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from typing import Dict

def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config,
) -> dict:
    """
    Generic PyTorch training loop with early stopping and checkpointing.
    Works for any model that outputs predictions given inputs.

    Args:
        model: torch.nn.Module - model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: configuration object with:
            - device
            - lr
            - num_epochs
            - optimizer ("adam" or "sgd")
            - early_stopping_patience
            - early_stopping_delta
            - results_path

    Returns:
        dict: final metrics (train_mse, train_mae, val_mse, val_mae)
    """

    model = model.to(config.device)
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
                torch.save(model.state_dict(), config.results_path / "best_model.pth")
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
    best_model_path = config.results_path / "best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=config.device))
        model.to(config.device)

    logging.info("Training complete.")
    return {
        "train_mse": avg_train_mse,
        "train_mae": avg_train_mae,
        "val_mse": avg_val_mse,
        "val_mae": avg_val_mae,
    }



def test_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config,
) -> Dict[str, float]:
    """
    Generic evaluation function for any PyTorch model.
    Computes MSE and MAE over the given test set.

    Args:
        model: torch.nn.Module - trained model to evaluate.
        test_loader: DataLoader - test dataset loader.
        device: torch.device - device to run evaluation on.
        config: configuration object with:
            - results_path (optional, for loading a checkpoint)
            - any other relevant fields (ignored by this function).

    Returns:
        dict: {"test_mse": float, "test_mae": float}
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
