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
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
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
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss


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
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        
        self.logger = logging.getLogger("main")
        
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        """
        Fit the DeepSurv model.
        
        Parameters:
            X: (n_samples, n_features) numpy array of features
            y: Structured array with 'event' and 'time' fields (scikit-survival format)
            X_val: Optional validation features
            y_val: Optional validation labels
            verbose: Whether to print training progress
        """
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
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Training loop
        self.network.train()
        for epoch in range(self.epochs):
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
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            if verbose and (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        # After training, estimate baseline hazard using Breslow estimator
        self._estimate_baseline_hazard(X, y)
        
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
        
        # Extract times and events
        times = y['time'].astype(np.float32)
        events = y['event'].astype(bool)
        
        # Get unique event times (sorted)
        unique_times = np.unique(times[events])
        
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
            X_tensor = torch.FloatTensor(X).to(self.device)
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
            if times is None:
                # Use original baseline times
                surv_prob = np.power(self.baseline_survival_, np.exp(risk_scores[i]))
                survival_functions.append((self.unique_times_.copy(), surv_prob))
            else:
                # Use interpolated baseline
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
