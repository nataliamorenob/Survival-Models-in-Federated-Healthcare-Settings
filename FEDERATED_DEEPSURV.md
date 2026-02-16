# Federated DeepSurv Implementation

## Overview

This implementation provides **federated DeepSurv using local risk set approximation** - the most common approach in research papers for federated Cox proportional hazards models.

## ⚠️ Important Limitations

### Mathematical Bias

**This approach is NOT equivalent to centralized DeepSurv** due to the non-separable nature of the Cox partial likelihood loss function.

#### The Problem

The Cox loss requires computing risk sets R(t_i) = {patients alive at time t_i}:

```
L_Cox = -Σ[h(x_i) - log(Σ_{j∈R(t_i)} exp(h(x_j)))]
```

In **centralized training**, R(t_i) contains ALL patients across ALL hospitals.

In **federated training with local risk sets**, R_A(t_i) contains ONLY Hospital A's patients.

**This introduces systematic bias** because:
- Each hospital computes gradients using incomplete risk sets
- FedAvg aggregates these biased gradients
- The aggregated gradient ≠ true gradient from global risk sets

### Why Use This Approach?

Despite the bias, local risk set approximation is used because:
1. ✅ **Privacy preserving** - no data sharing required
2. ✅ **Simple to implement** - standard FedAvg workflow
3. ✅ **Computationally efficient** - no communication overhead
4. ✅ **Works in practice** - often achieves reasonable C-index
5. ✅ **Common in literature** - enables comparison with other papers

## How It Works

### 1. Local Training (Each Hospital)

```python
# Hospital A trains DeepSurv using ONLY Hospital A's patients
# Risk sets R(t) computed from Hospital A's data only
model.fit(X_train_A, y_train_A)  # Biased gradients

# Extract weights
weights_A = get_parameters(model)
```

### 2. Server Aggregation

```python
# Server performs weighted averaging (FedAvg)
weights_global = Σ(n_k/n_total * weights_k)
```

### 3. Distribution

```python
# Server sends global weights back to hospitals
set_parameters(model, weights_global)
```

### 4. Evaluation

**Important**: After receiving global weights, each hospital must:
1. Load global neural network weights
2. **Re-estimate baseline hazard** using local training data
3. Evaluate on local test set

The baseline hazard h₀(t) is **non-parametric** and cannot be averaged, so each hospital estimates it locally.

## Usage

### Configuration

```python
from config import Config

config = Config(
    model="DeepSurv",
    centers=[0, 1, 2, 3, 4, 5],  # 6 hospitals
    training_mode="federated",
    num_clients=6,
    num_rounds=10,
    num_epochs=20,  # Local epochs per round
    deepsurv_hidden_layers=[64, 32, 16],
    deepsurv_dropout=0.3,
    deepsurv_batch_norm=True,
    deepsurv_activation='ReLU',
    deepsurv_l2_reg=0.001,
    lr=0.001,
    batch_size=64,
)
```

### Running Federated DeepSurv

```python
from main import main

main(config)
```

### Expected Output

```
[Server] Round 1: Aggregating DeepSurv weights from 6 clients
[Client 0] Training DeepSurv for 20 epochs
[Client 1] Training DeepSurv for 20 epochs
...
[Client 0] Evaluation finished → C-index=0.7234, AUC=0.6891, IBS=0.1245
[Client 1] Evaluation finished → C-index=0.7156, AUC=0.6823, IBS=0.1312
...
[Server] Round 1 - Aggregated Metrics → C-index=0.7195, AUC=0.6857, IBS=0.1278
```

## Files Created

1. **`Training_Modes/Federated_Learning/clientDeepSurv.py`**
   - `FederatedDeepSurvClient` class
   - Handles local training and evaluation
   - Weight serialization/deserialization

2. **`Training_Modes/Federated_Learning/strategies.py`** (updated)
   - Added `DeepSurvFedAvg` strategy
   - Implements weight averaging for PyTorch models

3. **`main.py`** (updated)
   - Added DeepSurv client creation in `client_fn()`
   - Added DeepSurvFedAvg strategy selection

## Comparison with Centralized DeepSurv

| Aspect | Centralized | Federated (Local Risk Sets) |
|--------|-------------|----------------------------|
| **Data** | All data pooled | Data stays local |
| **Risk sets** | Global (all patients) | Local (hospital's patients only) |
| **Mathematical correctness** | ✅ Correct | ❌ Biased |
| **Privacy** | ❌ Data sharing required | ✅ Fully private |
| **C-index** | Higher (unbiased) | Lower (biased approximation) |
| **Typical gap** | Baseline | -0.02 to -0.05 |

## Thesis Recommendations

### 1. Experimental Setup

Compare three training modes:
- **Local**: Each hospital trains independently
- **Centralized**: Pool all data (oracle/upper bound)
- **Federated**: This implementation (privacy-preserving)

### 2. Discussion Points

In your thesis, discuss:

1. **Non-separability problem**: Explain why Cox loss is inherently incompatible with federated learning
2. **Local risk set bias**: Quantify the performance gap vs. centralized
3. **Privacy-utility tradeoff**: Federated achieves privacy but sacrifices accuracy
4. **Alternative approaches**: Mention discrete-time models as truly federated-friendly alternatives

### 3. Expected Results

You should observe:
- Centralized DeepSurv: **Highest C-index** (has global risk sets)
- Federated DeepSurv: **Medium C-index** (biased but learns from federated data)
- Local DeepSurv: **Lowest C-index** (small local datasets)

### 4. Comparison with FedSurF

| Model | Federated Approach | Bias | C-index (Expected) |
|-------|-------------------|------|-------------------|
| **FedSurF** | Tree aggregation | ✅ None | High |
| **DeepSurv** | Local risk sets | ❌ Biased | Medium |

FedSurF is mathematically correct for federated learning because trees are separable.
DeepSurv requires this biased approximation.

## Citation

When using this implementation, cite it as:

> "We implement federated DeepSurv using local risk set approximation, where each hospital computes Cox partial likelihood using only local patients. While this introduces bias compared to centralized training, it preserves privacy and enables distributed learning."

## Advanced: Fixing the Bias (Future Work)

To eliminate bias, you would need:
1. **Secure aggregation** of risk scores (privacy-preserving)
2. **Global risk set construction** at the server
3. **Modified loss computation** using communicated risk scores

This is significantly more complex and is beyond the scope of this implementation.

## Support

For questions or issues, refer to:
- `DEEPSURV_USAGE.md` - General DeepSurv usage
- `FLamby/` - Federated learning framework
- Flower documentation - https://flower.dev/
