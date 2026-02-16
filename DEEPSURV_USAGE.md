# DeepSurv Integration Guide

## Overview

DeepSurv has been successfully integrated into your pipeline! It uses the same data preparation and evaluation functions as RSF, but implements a deep learning approach to survival analysis.

## Key Differences from RSF

### Model Type
- **RSF**: Ensemble of decision trees (Random Forest)
- **DeepSurv**: Deep neural network with Cox proportional hazards loss

### Output
- **RSF**: Produces survival curves (probability of survival over time)
- **DeepSurv**: Produces risk scores (higher = higher risk of event)

### Evaluation Metrics
- **RSF**: C-index, IPCW C-index, AUC, IBS (all 4 metrics)
- **DeepSurv**: C-index, IPCW C-index (risk-based metrics only)

## Configuration

### Basic Usage

In your `config.py` or when creating a Config object:

```python
from config import Config

config = Config(
    model="DeepSurv",  # Change from "RSF" to "DeepSurv"
    training_mode="local",  # or "centralized" or "federated"
    num_epochs=100,  # Training epochs
    batch_size=64,
    lr=0.001,
)
```

### DeepSurv-Specific Parameters

Optional hyperparameters (will use defaults if not specified):

```python
config = Config(
    model="DeepSurv",
    
    # Network architecture
    deepsurv_hidden_layers=[64, 32, 16],  # List of hidden layer sizes
    
    # Regularization
    deepsurv_dropout=0.3,  # Dropout rate (0.0 to disable)
    deepsurv_l2_reg=0.001,  # L2 regularization coefficient
    
    # Normalization
    deepsurv_batch_norm=True,  # Use batch normalization
    
    # Activation function
    deepsurv_activation='ReLU',  # Options: 'ReLU', 'Tanh', 'SELU', 'LeakyReLU'
    
    # Training parameters (shared with other models)
    num_epochs=100,
    batch_size=64,
    lr=0.001,
)
```

### Architecture Examples

**Smaller network (faster training, less overfitting):**
```python
deepsurv_hidden_layers=[32, 16]
deepsurv_dropout=0.2
```

**Larger network (more capacity, may need more data):**
```python
deepsurv_hidden_layers=[128, 64, 32, 16]
deepsurv_dropout=0.4
```

**No regularization (use with caution):**
```python
deepsurv_dropout=0.0
deepsurv_l2_reg=0.0
deepsurv_batch_norm=False
```

## Running DeepSurv

### Local Training

```python
from config import Config
from Training_Modes.Local_Learning.local_run import run_local

config = Config(
    model="DeepSurv",
    training_mode="local",
    num_epochs=100,
    centers=[0],  # Single center for local training
)

metrics = run_local(config)
print(f"C-index: {metrics['C-index']:.4f}")
```

### Centralized Training

```python
from config import Config
from Training_Modes.Centralized_Learning.centralized_run import run_centralized

config = Config(
    model="DeepSurv",
    training_mode="centralized",
    num_epochs=100,
    centers=[0, 1, 2, 3, 4, 5],  # All centers combined
)

metrics = run_centralized(config)
```

### Federated Training

```python
from config import Config
from Training_Modes.Federated_Learning.federated_run import run_federated

config = Config(
    model="DeepSurv",
    training_mode="federated",
    num_rounds=10,  # Federated learning rounds
    num_epochs=5,   # Local epochs per round
    num_clients=6,
    centers=[0, 1, 2, 3, 4, 5],
)

metrics = run_federated(config)
```

## Data Format Compatibility

### Automatic Conversion

DeepSurv automatically handles your existing data format:

```python
# Your current format (structured array from scikit-survival)
X_train = np.array([[...], [...], ...])  # (n_samples, 39)
y_train = np.array(
    [(True, 123.5), (False, 456.2), ...],
    dtype=[('event', 'bool'), ('time', 'f8')]
)

# DeepSurv wrapper automatically converts this internally
model.fit(X_train, y_train)  # No changes needed!
```

### What Happens Internally

The DeepSurv wrapper:
1. Extracts `event` and `time` from the structured array
2. Sorts data by time (descending) as required by Cox loss
3. Converts to PyTorch tensors
4. Trains the neural network
5. Returns risk scores for evaluation

## Evaluation

DeepSurv uses the same evaluation pipeline as RSF:

```python
from utils import evaluate_deepsurv

metrics = evaluate_deepsurv(
    model,
    data={
        "X_test": X_test,
        "y_test": y_test,
        "y_train": y_train,
    },
    client_id=0,
    config=config,
)

print(f"C-index: {metrics['C-index']:.4f}")
print(f"IPCW C-index: {metrics['IPCW_C-index']:.4f}")
```

## Model Comparison

You can easily compare DeepSurv with RSF:

```python
from config import Config
from Training_Modes.Local_Learning.local_run import run_local

# Test RSF
config_rsf = Config(model="RSF", training_mode="local", centers=[0])
metrics_rsf = run_local(config_rsf)

# Test DeepSurv
config_ds = Config(model="DeepSurv", training_mode="local", centers=[0])
metrics_ds = run_local(config_ds)

# Compare
print(f"RSF C-index: {metrics_rsf['C-index']:.4f}")
print(f"DeepSurv C-index: {metrics_ds['C-index']:.4f}")
```

## Saving and Loading Models

```python
# After training
model.save_model("deepsurv_model.pth")

# Load later
from Models.DeepSurv import DeepSurv

model = DeepSurv(n_features=39, ...)
model.load_model("deepsurv_model.pth")

# Continue using
risk_scores = model.predict_risk(X_test)
```

## GPU Support

DeepSurv automatically uses GPU if available:

```python
import torch

# Check if GPU is available
print(f"GPU available: {torch.cuda.is_available()}")

# DeepSurv will automatically use GPU
config = Config(model="DeepSurv", ...)
# Training will use GPU if available
```

## Troubleshooting

### Issue: Loss is NaN

**Cause**: Learning rate too high or data not properly normalized

**Solution**:
```python
config.lr = 0.0001  # Reduce learning rate
config.deepsurv_l2_reg = 0.01  # Increase regularization
```

### Issue: Model not learning (C-index ~0.5)

**Cause**: Network too small, learning rate too low, or not enough epochs

**Solution**:
```python
config.deepsurv_hidden_layers = [128, 64, 32]  # Larger network
config.lr = 0.001  # Reasonable learning rate
config.num_epochs = 200  # More epochs
```

### Issue: Overfitting (train good, test poor)

**Cause**: Not enough regularization

**Solution**:
```python
config.deepsurv_dropout = 0.5  # Increase dropout
config.deepsurv_l2_reg = 0.01  # Increase L2 regularization
```

## Next Steps

1. **Hyperparameter Tuning**: Experiment with different architectures and regularization
2. **Compare Models**: Run both RSF and DeepSurv to see which performs better
3. **Federated Learning**: Test DeepSurv in federated setting (requires additional implementation)
4. **Custom Metrics**: Add custom evaluation metrics if needed

## Technical Details

### Data Flow

```
Original Data (FLamby)
    ↓
DatasetManager
    ↓
Structured Arrays (X_train, y_train)
    ↓
DeepSurv.fit() → Internal conversion to tensors
    ↓
Neural Network Training
    ↓
DeepSurv.predict_risk() → Risk scores
    ↓
evaluate_deepsurv() → Metrics
```

### Network Architecture

```
Input (39 features)
    ↓
[Optional: Dropout]
Linear Layer → Hidden[0] neurons
BatchNorm (if enabled)
Activation (ReLU/Tanh/SELU)
    ↓
[Repeat for each hidden layer]
    ↓
Linear Layer → 1 output (risk score)
```

### Loss Function

Cox Partial Likelihood Loss + L2 Regularization:
```
L = -Σ[h_i - log(Σ exp(h_j))] / N_events + λ||W||²
```

where:
- h_i: risk score for event i
- Sum over events that occurred
- Denominator sums over samples at risk at time i
- λ: L2 regularization coefficient
