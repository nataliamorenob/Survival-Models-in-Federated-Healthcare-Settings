"""Linear CoxPH model aligned with the FLamby TCGA-BRCA baseline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Models.DeepSurv import NegativeLogLikelihood


class CoxPHNetwork(nn.Module):
    """FLamby-style CoxPH baseline: a single linear layer."""

    def __init__(self, n_features):
        super().__init__()
        self.fc = nn.Linear(n_features, 1)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


class CoxPHModel:
    """
    Linear Cox proportional hazards model trained with the same loss as DeepSurv.

    The architecture follows FLamby's TCGA-BRCA baseline, while training,
    baseline hazard estimation, and survival prediction mirror the existing
    DeepSurv wrapper so evaluation stays identical across both models.
    """

    def __init__(
        self,
        n_features,
        lr=0.001,
        l2_reg=0.0,
        epochs=100,
        batch_size=32,
        random_state=None,
        device=None,
        num_updates_per_round=None,
    ):
        self.n_features = n_features
        self.lr = lr
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.num_updates_per_round = num_updates_per_round
        self.model_name = "CoxPH"

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.network = CoxPHNetwork(n_features).to(self.device)
        self.criterion = NegativeLogLikelihood({})
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.lr,
            weight_decay=self.l2_reg,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )
        self.logger = logging.getLogger("main")

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        verbose=True,
        client_id=None,
        log_file=None,
        proximal_mu=None,
        global_weights=None,
    ):
        file_handler = None
        if log_file:
            import os

            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.logger.addHandler(file_handler)

        if not (isinstance(y, np.ndarray) and y.dtype.names):
            raise ValueError("y must be a structured array with 'event' and 'time' fields")

        events = y["event"].astype(np.float32)
        times = y["time"].astype(np.float32)

        sort_idx = np.argsort(times)[::-1]
        X_sorted = X[sort_idx]
        times_sorted = times[sort_idx]
        events_sorted = events[sort_idx]

        X_tensor = torch.FloatTensor(X_sorted).to(self.device)
        times_tensor = torch.FloatTensor(times_sorted).reshape(-1, 1).to(self.device)
        events_tensor = torch.FloatTensor(events_sorted).reshape(-1, 1).to(self.device)

        dataset = TensorDataset(X_tensor, times_tensor, events_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        val_dataloader = None
        if X_val is not None and y_val is not None:
            val_events = y_val["event"].astype(np.float32)
            val_times = y_val["time"].astype(np.float32)
            val_sort_idx = np.argsort(val_times)[::-1]

            X_val_tensor = torch.FloatTensor(X_val[val_sort_idx]).to(self.device)
            val_times_tensor = torch.FloatTensor(val_times[val_sort_idx]).reshape(-1, 1).to(self.device)
            val_events_tensor = torch.FloatTensor(val_events[val_sort_idx]).reshape(-1, 1).to(self.device)

            val_dataset = TensorDataset(X_val_tensor, val_times_tensor, val_events_tensor)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=len(X_val_tensor),
                shuffle=False,
                drop_last=False,
            )

        num_updates_per_epoch = len(X_sorted) / float(self.batch_size)
        if self.num_updates_per_round is not None:
            effective_epochs = int(np.ceil(self.num_updates_per_round / num_updates_per_epoch))
            print(
                f"[{self.model_name}] Using fixed updates approach: "
                f"{self.num_updates_per_round} updates = {effective_epochs} epochs"
            )
        else:
            effective_epochs = self.epochs
            print(f"[{self.model_name}] Using fixed epochs approach: {effective_epochs} epochs")

        best_val_loss = float("inf")
        best_epoch = 0
        patience = 5
        patience_counter = 0
        best_state_dict = None

        self.network.train()
        print(f"[{self.model_name}] Starting training for {effective_epochs} epochs...")

        for epoch in range(effective_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_t, batch_e in dataloader:
                self.optimizer.zero_grad()
                risk_pred = self.network(batch_X)
                loss = self.criterion(risk_pred, batch_t, batch_e, self.network)

                if proximal_mu is not None and global_weights is not None:
                    proximal_term = 0.0
                    for local_param, global_param in zip(self.network.parameters(), global_weights):
                        proximal_term += torch.sum((local_param - global_param) ** 2)
                    loss = loss + (proximal_mu / 2.0) * proximal_term

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            if val_dataloader is not None:
                self.network.eval()
                val_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for batch_X, batch_t, batch_e in val_dataloader:
                        risk_pred = self.network(batch_X)
                        loss = self.criterion(risk_pred, batch_t, batch_e, self.network)
                        val_loss += loss.item()
                        val_batches += 1

                avg_val_loss = val_loss / max(val_batches, 1)
                self.network.train()
                self.scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in self.network.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if verbose:
                    prefix = f"[Client {client_id} {self.model_name}]" if client_id is not None else f"[{self.model_name}]"
                    log_msg = (
                        f"{prefix} Epoch {epoch + 1}/{effective_epochs} | "
                        f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
                    )
                    print(log_msg)
                    if log_file:
                        self.logger.info(log_msg)
                    elif (epoch + 1) % 10 == 0:
                        self.logger.info(log_msg)

                if patience_counter >= patience:
                    prefix = f"[Client {client_id} {self.model_name}]" if client_id is not None else f"[{self.model_name}]"
                    stop_msg = (
                        f"{prefix} Early stopping at epoch {epoch + 1}. "
                        f"Best epoch: {best_epoch + 1} | Best val loss: {best_val_loss:.4f}"
                    )
                    print(stop_msg)
                    if log_file:
                        self.logger.info(stop_msg)
                    if best_state_dict is not None:
                        self.network.load_state_dict(
                            {key: value.to(self.device) for key, value in best_state_dict.items()}
                        )
                    break
            elif verbose and (epoch + 1) % 5 == 0:
                prefix = f"[Client {client_id} {self.model_name}]" if client_id is not None else f"[{self.model_name}]"
                log_msg = f"{prefix} Epoch {epoch + 1}/{effective_epochs} | Train Loss: {avg_loss:.4f}"
                print(log_msg)
                if (epoch + 1) % 10 == 0:
                    self.logger.info(log_msg)

        if val_dataloader is not None and best_state_dict is not None:
            msg = (
                f"[{self.model_name}] Training finished. Best epoch: {best_epoch + 1} | "
                f"Best val loss: {best_val_loss:.4f}"
            )
            print(msg)
            if verbose:
                self.logger.info(msg)
        else:
            msg = f"[{self.model_name}] Training finished. Final train loss: {avg_loss:.4f}"
            print(msg)
            if verbose:
                self.logger.info(msg)

        self._estimate_baseline_hazard(X, y)

        if file_handler:
            self.logger.removeHandler(file_handler)
            file_handler.close()

        return self

    def _estimate_baseline_hazard(self, X, y):
        risk_scores = np.clip(self.predict_risk(X), -50.0, 50.0)
        times = y["time"].astype(np.float32)
        events = y["event"].astype(bool)
        unique_times = np.unique(times[events])

        if len(unique_times) == 0:
            self.logger.warning("No events in training data; baseline hazard cannot be estimated.")
            self.baseline_hazard_ = None
            self.baseline_survival_ = None
            self.unique_times_ = None
            return

        baseline_hazard = np.zeros(len(unique_times), dtype=np.float64)
        exp_risk_scores = np.exp(risk_scores)

        for idx, time_point in enumerate(unique_times):
            n_events = np.sum((times == time_point) & events)
            at_risk = times >= time_point
            denominator = np.sum(exp_risk_scores[at_risk])
            if denominator > 0:
                baseline_hazard[idx] = n_events / denominator

        cumulative_baseline_hazard = np.cumsum(baseline_hazard)
        self.unique_times_ = unique_times
        self.baseline_hazard_ = cumulative_baseline_hazard
        self.baseline_survival_ = np.exp(-cumulative_baseline_hazard)

    def reset_optimizer_state(self):
        self.optimizer.state = {}
        self.logger.debug("Optimizer state cleared to free memory")

    def predict_risk(self, X):
        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.ascontiguousarray(X)).to(self.device)
            risk_pred = self.network(X_tensor)
        return risk_pred.cpu().numpy().flatten()

    def predict_survival_function(self, X, times=None):
        if not hasattr(self, "baseline_survival_") or self.baseline_survival_ is None:
            self.logger.error("Baseline hazard not estimated. Call fit() before prediction.")
            return None

        risk_scores = np.clip(self.predict_risk(X), -50.0, 50.0)

        if times is None:
            times = self.unique_times_
            baseline_surv_at_times = self.baseline_survival_
        else:
            from scipy.interpolate import interp1d

            interpolation = interp1d(
                self.unique_times_,
                self.baseline_survival_,
                kind="previous",
                bounds_error=False,
                fill_value=(1.0, self.baseline_survival_[-1]),
            )
            baseline_surv_at_times = interpolation(times)

        survival_functions = []
        for risk_score in risk_scores:
            survival_prob = np.power(baseline_surv_at_times, np.exp(risk_score))
            survival_functions.append((times.copy(), survival_prob))

        return survival_functions

    def get_params(self):
        return {
            "n_features": self.n_features,
            "lr": self.lr,
            "l2_reg": self.l2_reg,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }
