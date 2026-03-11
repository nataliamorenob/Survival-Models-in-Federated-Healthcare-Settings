# FedSurF++ Implementation Guide

## Overview

This guide explains the implementation of **FedSurF++** with C-Index based tree sampling, following the paper ["FedSurF++: Enhanced Federated Survival Forest with C-Index Sampling"](https://arxiv.org/...).

## Algorithm Description

FedSurF++ is a federated survival analysis algorithm that combines Random Survival Forests (RSF) with federated learning. It consists of three main phases:

### Phase 1: Local Training
Each client k trains a local Random Survival Forest Mk with Tk trees using their local data Dk.

### Phase 2: Tree Assignment
The server determines how many trees T'k each client should contribute to the global forest. This is done proportionally to the client's dataset size Nk, similar to FedAvg's weighted aggregation.

Algorithm (lines 4-7 from paper):
```
For i = 1 to T (total global trees):
    Sample client k with probability proportional to Nk
    Increment T'k
```

### Phase 3: Tree Sampling (C-Index)
Each client evaluates their local trees using the Concordance Index (C-Index) on a validation set, then samples T'k trees proportionally to their C-Index scores.

For client k:
1. Evaluate each tree j: compute C-Index_j on validation data
2. Build sampling distribution: p_j = C-Index_j / Σ(C-Index_j)
3. Sample T'k trees without replacement using p_j

## Implementation Components

### 1. Model: `Models/RSF_FedSurF.py`

**Class:** `RSFFedSurFPlus`

Wraps scikit-survival's `RandomSurvivalForest` with FedSurF-specific functionality:
- `fit(X, y)`: Train local RSF
- `estimators_`: Access trained decision trees
- `set_trees(trees, n_features, init_times)`: Load global federated forest
- `predict_survival_function_fedsurf(X)`: Predict with tree averaging

### 2. Client: `Training_Modes/Federated_Learning/clientRSFFedSurF.py`

**Class:** `FederatedRSFFedSurFClient`

Implements the client-side logic:

**Round 1 (Training & Evaluation):**
1. Train local RSF with `n_trees_local` trees
2. Evaluate each tree on validation set using C-Index
3. Send ALL trees + C-Index scores to server

**Round 2+ (Evaluation):**
1. Load global forest from server
2. Evaluate on test set using `evaluate_rsf()`

### 3. Strategy: `Training_Modes/Federated_Learning/strategies.py`

**Class:** `FedSurFPlusPlus`

Implements the server-side aggregation:

**Round 1 (aggregate_fit):**
1. Collect all trees and metadata from clients
2. **Tree Assignment:** Determine T'k for each client (proportional to Nk)
3. **Tree Sampling:** Sample T'k trees from each client (proportional to C-Index)
4. Build global forest with exactly `n_trees_federated` trees

**Round 2+ (configure_evaluate):**
- Send global forest to all clients for evaluation

### 4. Configuration

The configuration in `config.py` includes:

```python
# RSF FedSurF++ parameters
n_trees_local: int = 200          # Trees trained per client
n_trees_federated: int = 80       # Total trees in global forest
min_samples_split: int = 5        # Min samples for node split
min_samples_leaf: int = 10        # Min samples at leaf
random_state: int = 42            # Random seed
```

## Usage

### Basic Example

```python
from src.config import Config
from src.main import main

# Configure FedSurF++ experiment
user_config = Config(
    model="RSF_FedSurF",
    centers=[0, 1, 2, 3, 4],
    training_mode="federated",
    num_clients=5,
    strategy="FedSurFPlusPlus",
    n_trees_local=200,
    n_trees_federated=80,
    eval_grid_mode="global"
)

# Run experiment
main(user_config)
```

### Command Line Execution

```bash
# Activate environment
source venv/bin/activate

# Run FedSurF++ experiment
python src/main.py
```

## Key Differences from Original FedSurvForest

| Aspect | FedSurvForest | FedSurF++ |
|--------|---------------|-----------|
| Tree Assignment | Random sampling weighted by dataset size | Explicit iterative assignment (Algorithm 1, lines 4-7) |
| Tree Sampling | Global C-Index based | Client-specific C-Index based |
| Implementation | Combined assignment + sampling | Separate two-phase approach |
| Follows Paper | Partial | Complete Algorithm 1 |

## Evaluation Metrics

The implementation uses `evaluate_rsf()` from `utils.py` which computes:

1. **C-Index (Concordance Index)**: Primary metric for ranking predictions
2. **AUC (Area Under Curve)**: Time-dependent discriminative ability
3. **IBS (Integrated Brier Score)**: Calibration and prediction accuracy

## Files Created/Modified

**New Files:**
- `src/Models/RSF_FedSurF.py` - RSF model wrapper
- `src/Training_Modes/Federated_Learning/clientRSFFedSurF.py` - FedSurF++ client
- `src/Training_Modes/Federated_Learning/strategies.py` - Added `FedSurFPlusPlus` strategy

**Modified Files:**
- `src/model_manager.py` - Added RSF_FedSurF model recognition
- `src/main.py` - Added client factory and strategy configuration

## Logging

The implementation provides detailed logging:

```
[FedSurF++] Client 0: n_samples=120, n_trees=200, mean_C-index=0.6234
[FedSurF++] Tree Assignment Phase: total_samples=578, clients=[0,1,2,3,4]
[FedSurF++] Tree Assignment Result: Client 0: 15 | Client 1: 18 | ...
[FedSurF++] Client 0: sampled 15 trees (C-index range: [0.58, 0.67])
[FedSurF++] Global forest built: 80 trees (target: 80)
```

## Hyperparameter Tuning

Key hyperparameters to tune:

1. **n_trees_local** (100-500): More trees = better local model, longer training
2. **n_trees_federated** (50-200): More trees = better global model, higher communication
3. **min_samples_split** (5-20): Controls tree depth and overfitting
4. **min_samples_leaf** (5-15): Prevents overly specific leaf nodes

## Reproducibility

Seeds are set for reproducibility:
- Global seed: `config.random_state`
- Client seed: `config.random_state + client_id`

This ensures deterministic results across runs while maintaining client diversity.

## Troubleshooting

**Issue:** "Global forest has fewer trees than target"
- **Cause:** Clients don't have enough high-quality trees
- **Solution:** Increase `n_trees_local` or improve data quality

**Issue:** Low C-Index scores
- **Cause:** Insufficient tree diversity or poor validation set
- **Solution:** Check data splits, increase tree parameters

**Issue:** Memory errors
- **Cause:** Too many trees or large forests
- **Solution:** Reduce `n_trees_local` or use smaller batches

## References

- Original FedSurF paper
- FedSurF++ paper (this implementation)
- scikit-survival documentation
- Flower FL framework documentation

## Contact

For questions or issues with this implementation, please refer to the project README or open an issue.
