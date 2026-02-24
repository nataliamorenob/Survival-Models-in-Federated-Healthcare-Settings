# Models/DeepSurv.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import logging


class Regularization(object):
    def __init__(self, order, weight_decay):
        """The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        """
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        """Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        """
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


class DeepSurvNetwork(nn.Module):
    """The module class performs building network according to config"""
    def __init__(self, config):
        super(DeepSurvNetwork, self).__init__()
        # parses parameters of network from configuration
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        # builds network
        self.model = self._build_network()

    def _build_network(self):
        """Performs building networks according to parameters"""
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None:  # adds dropout layer
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            linear = nn.Linear(self.dims[i], self.dims[i+1])
            # Initialize weights properly for stable training
            nn.init.xavier_normal_(linear.weight)
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)
            if self.norm:  # adds batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            # adds activation layer
            layers.append(eval('nn.{}()'.format(self.activation)))
        # builds sequential network
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)


class NegativeLogLikelihood(nn.Module):
    def __init__(self, config):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        # Clamp risk predictions to prevent overflow in exp()
        risk_pred = torch.clamp(risk_pred, min=-50, max=50)
        
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        
        # Add numerical stability
        log_loss = torch.exp(risk_pred) * mask
        sum_mask = torch.sum(mask, dim=0)
        sum_mask = torch.clamp(sum_mask, min=1.0)  # Prevent division by zero
        
        log_loss = torch.sum(log_loss, dim=0) / sum_mask
        log_loss = torch.clamp(log_loss, min=1e-7)  # Prevent log(0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        
        # Calculate negative log likelihood
        num_events = torch.sum(e)
        if num_events == 0:
            return torch.tensor(0.0, requires_grad=True)  # No events, return zero loss
        
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / num_events
        l2_loss = self.reg(model)
        
        total_loss = neg_log_loss + l2_loss
        
        # Check for NaN/inf and return a large finite value instead
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("[WARNING] Loss is NaN/Inf, returning large finite value")
            return torch.tensor(1000.0, requires_grad=True)
        
        return total_loss


class DeepSurv:
    """
    PyTorch DeepSurv wrapper compatible with the existing pipeline.
    
    This class wraps the DeepSurvNetwork to provide an interface similar to
    scikit-learn models, making it compatible with the RSF evaluation pipeline.
    """
    
    def __init__(
        self, 
        n_features,
        hidden_layers=[64, 32, 16],
        dropout=0.3,
        batch_norm=True,
        activation='ReLU',
        lr=0.001,
        l2_reg=0.0,
        epochs=100,
        batch_size=64,
        random_state=None,
        device=None
    ):
        """
        Initialize DeepSurv model.
        
        Parameters:
            n_features: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate (None to disable)
            batch_norm: Whether to use batch normalization
            activation: Activation function name ('ReLU', 'Tanh', 'SELU', etc.)
            lr: Learning rate
            l2_reg: L2 regularization coefficient
            epochs: Number of training epochs
            batch_size: Batch size for training
            random_state: Random seed for reproducibility
            device: torch device ('cuda' or 'cpu')
        """
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation
        self.lr = lr
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            
        # Build network configuration
        self.config = {
            'dims': [n_features] + hidden_layers + [1],  # +1 for output layer
            'drop': dropout,
            'norm': batch_norm,
            'activation': activation,
            'l2_reg': l2_reg
        }
        
        # Initialize network
        self.network = DeepSurvNetwork(self.config).to(self.device)
        self.criterion = NegativeLogLikelihood(self.config)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.lr,
            weight_decay=0.0  # L2 reg handled in loss function
        )
        
        # Learning rate scheduler with warm-up for stability
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=False
        )
        
        self.logger = logging.getLogger("main")
        
    def fit(self, X, y, X_val=None, y_val=None, verbose=True, client_id=None, log_file=None):
        """
        Fit the DeepSurv model.
        
        Parameters:
            X: (n_samples, n_features) numpy array of features
            y: Structured array with 'event' and 'time' fields (scikit-survival format)
            X_val: Optional validation features
            y_val: Optional validation labels
            verbose: Whether to print training progress
            client_id: Optional client identifier for federated learning (added to log prefix)
            log_file: Optional file path to write training logs (useful for federated learning)
        """
        # Setup file logging if specified
        file_handler = None
        if log_file:
            import os
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(file_handler)
        # Convert structured array to separate arrays
        if isinstance(y, np.ndarray) and y.dtype.names:
            events = y['event'].astype(np.float32)
            times = y['time'].astype(np.float32)
        else:
            raise ValueError("y must be a structured array with 'event' and 'time' fields")
            
        # Sort by time (descending) - required for Cox loss
        sort_idx = np.argsort(times)[::-1]
        X_sorted = X[sort_idx]
        times_sorted = times[sort_idx]
        events_sorted = events[sort_idx]
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_sorted).to(self.device)
        times_tensor = torch.FloatTensor(times_sorted).reshape(-1, 1).to(self.device)
        events_tensor = torch.FloatTensor(events_sorted).reshape(-1, 1).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, times_tensor, events_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            drop_last=True  # Skip last incomplete batch to avoid BatchNorm errors
        )
        
        # Prepare validation data if provided
        val_dataloader = None
        if X_val is not None and y_val is not None:
            val_events = y_val['event'].astype(np.float32)
            val_times = y_val['time'].astype(np.float32)
            val_sort_idx = np.argsort(val_times)[::-1]
            
            X_val_sorted = X_val[val_sort_idx]
            val_times_sorted = val_times[val_sort_idx]
            val_events_sorted = val_events[val_sort_idx]
            
            X_val_tensor = torch.FloatTensor(X_val_sorted).to(self.device)
            val_times_tensor = torch.FloatTensor(val_times_sorted).reshape(-1, 1).to(self.device)
            val_events_tensor = torch.FloatTensor(val_events_sorted).reshape(-1, 1).to(self.device)
            
            val_dataset = TensorDataset(X_val_tensor, val_times_tensor, val_events_tensor)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )
        
        # Early stopping parameters
        best_val_loss = float('inf')
        best_epoch = 0
        patience = 5  # Stop if no improvement for 5 epochs
        patience_counter = 0
        best_state_dict = None
        
        # Training loop
        self.network.train()
        print(f"[DeepSurv] Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            # Training phase
            epoch_loss = 0
            n_batches = 0
            
            for batch_X, batch_t, batch_e in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                risk_pred = self.network(batch_X)
                
                # Calculate loss
                loss = self.criterion(risk_pred, batch_t, batch_e, self.network)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            # Validation phase
            if val_dataloader is not None:
                self.network.eval()
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_X, batch_t, batch_e in val_dataloader:
                        risk_pred = self.network(batch_X)
                        loss = self.criterion(risk_pred, batch_t, batch_e, self.network)
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                self.network.train()
                
                # Update learning rate based on validation loss
                self.scheduler.step(avg_val_loss)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best model state
                    best_state_dict = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                else:
                    patience_counter += 1
                
                # Log every epoch with both train and val loss
                if verbose:
                    prefix = f"[Client {client_id} DeepSurv]" if client_id is not None else "[DeepSurv]"
                    log_msg = f"{prefix} Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
                    print(log_msg)
                    if log_file:
                        self.logger.info(log_msg)
                    elif (epoch + 1) % 10 == 0:
                        self.logger.info(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                
                # Early stopping
                if patience_counter >= patience:
                    prefix = f"[Client {client_id} DeepSurv]" if client_id is not None else "[DeepSurv]"
                    stop_msg = f"{prefix} Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1} | Best val loss: {best_val_loss:.4f}"
                    print(stop_msg)
                    if log_file:
                        self.logger.info(stop_msg)
                    if verbose:
                        self.logger.info(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1} | Best val loss: {best_val_loss:.4f}")
                    # Restore best model
                    self.network.load_state_dict({k: v.to(self.device) for k, v in best_state_dict.items()})
                    break
            else:
                # No validation - just log training loss
                if verbose and (epoch + 1) % 5 == 0:
                    prefix = f"[Client {client_id} DeepSurv]" if client_id is not None else "[DeepSurv]"
                    print(f"{prefix} Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_loss:.4f}")
                    if (epoch + 1) % 10 == 0:
                        self.logger.info(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_loss:.4f}")
        
        if val_dataloader is not None and best_state_dict is not None:
            prefix = f"[Client {client_id} DeepSurv]" if client_id is not None else "[DeepSurv]"
            msg = f"{prefix} Training finished. Best epoch: {best_epoch+1} | Best val loss: {best_val_loss:.4f}"
            print(msg)
            if verbose:
                self.logger.info(msg)
        else:
            msg = f"[DeepSurv] Training finished. Final train loss: {avg_loss:.4f}"
            print(msg)
            if verbose:
                self.logger.info(msg)
        
        # After training, estimate baseline hazard using Breslow estimator
        self._estimate_baseline_hazard(X, y)
        
        # Clean up file handler if it was added
        if file_handler:
            self.logger.removeHandler(file_handler)
            file_handler.close()
        
        return self
    
    def _estimate_baseline_hazard(self, X, y):
        """
        Estimate baseline cumulative hazard using Breslow estimator.
        
        This is needed to convert risk scores into survival probabilities.
        
        Parameters:
            X: Training features
            y: Training labels (structured array)
        """
        # Get risk scores for training data
        risk_scores = self.predict_risk(X)
        
        # DEBUG: Check risk score distribution
        print(f"[DeepSurv] Risk scores - min: {risk_scores.min():.4f}, max: {risk_scores.max():.4f}, mean: {risk_scores.mean():.4f}, std: {risk_scores.std():.4f}")
        
        # Extract times and events
        times = y['time'].astype(np.float32)
        events = y['event'].astype(bool)
        
        # Get unique event times (sorted)
        unique_times = np.unique(times[events])
        
        print(f"[DeepSurv] Baseline hazard estimation - {len(unique_times)} unique event times, {events.sum()} total events")
        
        if len(unique_times) == 0:
            self.logger.warning("No events in training data - cannot estimate baseline hazard")
            self.baseline_hazard_ = None
            self.baseline_survival_ = None
            return
        
        # Breslow estimator for baseline cumulative hazard
        baseline_hazard = np.zeros(len(unique_times))
        
        for i, t in enumerate(unique_times):
            # Number of events at time t
            d_t = np.sum((times == t) & events)
            
            # Risk set at time t (all samples with time >= t)
            at_risk = times >= t
            
            # Sum of exp(risk) for samples at risk
            risk_sum = np.sum(np.exp(risk_scores[at_risk]))
            
            if risk_sum > 0:
                baseline_hazard[i] = d_t / risk_sum
        
        # Cumulative baseline hazard
        cumulative_baseline_hazard = np.cumsum(baseline_hazard)
        
        # Store for later use
        self.unique_times_ = unique_times
        self.baseline_hazard_ = cumulative_baseline_hazard
        
        # Baseline survival function S_0(t) = exp(-H_0(t))
        self.baseline_survival_ = np.exp(-cumulative_baseline_hazard)
    
    def reset_optimizer_state(self):
        """
        Reset optimizer state to free memory.
        Useful in federated learning between rounds.
        """
        self.optimizer.state = {}
        self.logger.debug("Optimizer state cleared to free memory")
    
    def predict_risk(self, X):
        """
        Predict risk scores for samples.
        
        Parameters:
            X: (n_samples, n_features) numpy array of features
            
        Returns:
            risk_scores: (n_samples,) array of predicted risk scores
        """
        self.network.eval()
        with torch.no_grad():
            # Ensure array is contiguous and writable
            X_copy = np.ascontiguousarray(X)
            X_tensor = torch.FloatTensor(X_copy).to(self.device)
            risk_pred = self.network(X_tensor)
            risk_scores = risk_pred.cpu().numpy().flatten()
        return risk_scores
    
    def predict_survival_function(self, X, times=None):
        """
        Predict survival functions using baseline hazard and risk scores.
        
        Uses the formula: S(t|x) = S_0(t)^exp(h(x))
        where S_0(t) is the baseline survival and h(x) is the risk score.
        
        Parameters:
            X: (n_samples, n_features) numpy array of features
            times: Optional array of time points. If None, uses training event times.
            
        Returns:
            List of tuples (times, survival_probabilities) for each sample,
            matching the format expected by evaluate_rsf.
        """
        if not hasattr(self, 'baseline_survival_') or self.baseline_survival_ is None:
            self.logger.error(
                "Baseline hazard not estimated. Call fit() first or "
                "ensure training data has events."
            )
            return None
        
        # Get risk scores
        risk_scores = self.predict_risk(X)
        
        # Use training event times if not specified
        if times is None:
            times = self.unique_times_
            baseline_surv_at_times = self.baseline_survival_
        else:
            # Need to interpolate baseline survival to requested times
            from scipy.interpolate import interp1d
            f = interp1d(
                self.unique_times_,
                self.baseline_survival_,
                kind='previous',
                bounds_error=False,
                fill_value=(1.0, self.baseline_survival_[-1])
            )
            baseline_surv_at_times = f(times)
        
        # Compute survival function for each sample
        # S(t|x) = S_0(t)^exp(h(x))
        n_samples = len(X)
        survival_functions = []
        
        for i in range(n_samples):
            surv_prob = np.power(baseline_surv_at_times, np.exp(risk_scores[i]))
            survival_functions.append((times.copy(), surv_prob))
        
        return survival_functions
    
    def get_params(self):
        """Return model parameters (for compatibility)."""
        return {
            'n_features': self.n_features,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'activation': self.activation,
            'lr': self.lr,
            'l2_reg': self.l2_reg,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'random_state': self.random_state
        }
    
    def save_model(self, filepath):
        """Save model weights and configuration."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'params': self.get_params()
        }, filepath)
        
    def load_model(self, filepath):
        """Load model weights and configuration."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
