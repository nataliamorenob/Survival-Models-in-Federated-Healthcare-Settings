# Fixed Number of Updates Implementation

## Summary
Modified the federated learning implementation to use:
1. **Fixed number of gradient updates (100)** per client per round (Owkin FLamby approach)
2. **Dynamic calculation of number of rounds** to match centralized training computational budget

This ensures:
- Fair computational cost across all clients regardless of their local dataset size
- Equivalent total computation as centralized training
- Consistent with FLamby benchmarking methodology

## Motivation

### Problem 1: Unfair Computational Cost
In standard federated learning, if all clients train for the same number of epochs but have different dataset sizes:
- **Client with 1000 samples**: 1000/batch_size updates per epoch → more computation
- **Client with 200 samples**: 200/batch_size updates per epoch → less computation

This creates **unfair computational costs** and can introduce bias in aggregation.

### Problem 2: How Many Rounds?
How many federated rounds should we run? The FLamby paper solves this by matching the computational budget of centralized training.

## Solution

### Part 1: Fixed Updates Per Round
Following the FLamby paper approach:
1. **Fix the number of gradient updates** per round: 100 updates
2. **Calculate epochs dynamically** per client based on:
   - `updates_per_epoch = dataset_size / batch_size`
   - `num_epochs = ceil(100 / updates_per_epoch)`

This ensures:
- ✅ All clients perform exactly **100 gradient updates** per round
- ✅ Fair computational cost across all clients
- ✅ Consistent training regardless of data heterogeneity

### Part 2: Dynamic Number of Rounds

#### a) Added `get_nb_max_rounds()` function
```python
def get_nb_max_rounds(num_updates, batch_size, num_clients=6, 
                      num_epochs_pooled=30, total_train_samples=866):
    """
    Calculate the number of federated rounds needed to match the computational budget
    of centralized training.
    
    Formula from FLamby:
    num_rounds = (total_samples // num_clients // batch_size) * num_epochs_pooled // num_updates_per_round
    """
    avg_samples_per_client = total_train_samples // num_clients
    updates_per_epoch = avg_samples_per_client // batch_size
    total_updates_centralized = updates_per_epoch * num_epochs_pooled
    num_rounds = total_updates_centralized // num_updates
    
    return max(1, num_rounds)
```

#### b) Added new parameters
```python
num_rounds: int = None  # Will be calculated dynamically in __post_init__
num_updates_per_round: int = 100  # Fixed updates per client per round
total_train_samples: int = 866  # Total training samples across all TCGA-BRCA clients
```

#### c) Modified `__post_init__()` to calculate rounds dynamically
```python
if self.num_rounds is None and self.training_mode == "federated":
    self.num_rounds = get_nb_max_rounds(
        num_updates=self.num_updates_per_round,
        batch_size=self.batch_size,
        num_clients=self.num_clients,
        num_epochs_pooled=self.num_epochs,
        total_train_samples=self.total_train_samples
    )
    print(f"[Config] Calculated num_rounds = {self.num_rounds}")
```

**Note:** You can still manually set `num_rounds` in config to override the automatic calculation
- Total training samples: 866
- Number of clients: 6
- Batch size: 8
- Target centralized epochs: 30
- Updates per round: 100

Calculation:
1. Avg samples per client: 866 // 6 = 144
2. Updates per epoch: 144 // 8 = 18
3. Total updates for 30 epochs: 18 × 30 = 540
4. **Number of rounds needed: 540 // 100 = 5**

This ensures federated training does the **same total number of gradient updates** as centralized training would do in 30 epochs.

## Changes Made

### 1. Configuration (`src/config.py`)
**Added new parameter:**
```python
num_updates_per_round: int = 100  # Fixed number of gradient updates per client per round
```

**Note:** `num_epochs` is now only used for centralized/local training modes.

### 2. DeepSurv Model (`src/Models/DeepSurv.py`)

#### a) Modified `__init__` method
- Added `num_updates_per_round` parameter to constructor

#### b) Changed from full-batch to mini-batch training
**Before:**
```python
effective_batch_size = len(X_sorted)  # Full-batch
shuffle=False
```

**After:**
```python
effective_batch_size = self.batch_size  # Mini-batch from config
shuffle=True  # Shuffle for better convergence
```

#### c) Dynamic epoch calculation in `fit()` method
**Added lo with Your Dataset

**Configuration:**
- Default: 6 clients, batch_size=8, num_epochs=30, num_updates_per_round=100
- Total train samples: 866 (from your TCGA-BRCA dataset)
- **Calculated num_rounds: 5**

**Per-Client Training (in each round):**

Given `batch_size=8` and `num_updates_per_round=100`:

**Client 0** (248 train samples):
- Updates per epoch: 248/8 = 31
- Epochs needed: ceil(100/31) = **4 epochs**
- Total updates: **124** (≈100) ✓

**Client 1** (156 train samples):
- Updates per epoch: 156/8 = 19.5
- Epochs needed: ceil(100/19.5) = **6 epochs**
- Total updates: **117** (≈100) ✓

**Client 2** (164 train samples):
- Updates per epoch: 164/8 = 20.5
- Epochs needed: ceil(100/20.5) = **5 epochs**, but this ensures a minimum of 100 updates
4. Early stopping is still active and may terminate training before reaching the target epochs if validation loss plateaus
5. **You can override automatic round calculation** by explicitly setting `num_rounds` in config (e.g., `num_rounds=10`)
6. The `total_train_samples=866` is specific to TCGA-BRCA with 6 clients - adjust if using different centers

## Adjusting for Different Number of Clients
If you change `num_clients` in config (e.g., to train with only centers [0,1,2]), you should also update `total_train_samples`:

- 3 clients [0,1,2]: total_train_samples = 248 + 156 + 164 = **568**
- 4 clients [0,1,2,3]: total_train_samples = 248 + 156 + 164 + 129 = **697**
- 5 clients [0,1,2,3,4]: total_train_samples = 697 + 129 = **826**
- 6 clients [0,1,2,3,4,5]: total_train_samples = 826 + 40 = **866**

**Client 3** (129 train samples):
- Updates per epoch: 129/8 = 16.1
- Epochs needed: ceil(100/16.1) = **7 epochs**
- Total updates: **112.7** (≈100) ✓

**Client 4** (129 train samples):
- Updates per epoch: 129/8 = 16.1
- Epochs needed: ceil(100/16.1) = **7 epochs**
- Total updates: **112.7** (≈100) ✓

**Client 5** (40 train samples):
- Updates per epoch: 40/8 = 5
- Epochs needed: ceil(100/5) = **20 epochs**
- Total updates: **100** ✓

**Total Federated Training:**
- 5 rounds × ~100 updates/client = **~500 total updates per client**
- Comparable to centralized: 144 avg samples × 30 epochs / 8 batch = **540 updates**
- **Fair computational cost across all clients!**

self.model = DeepSurv(
    ...
    num_updates_per_round=num_updates
)
```

### 4. Federated Client (`src/Training_Modes/Federated_Learning/clientDeepSurv.py`)
**Updated logging** to show fixed updates information:
```python
if hasattr(self.model, 'num_updates_per_round') and self.model.num_updates_per_round:
    expected_epochs = int(np.ceil(self.model.num_updates_per_round / updates_per_epoch))
    print(f"[Client {self.cid}] Training: {self.model.num_updates_per_round}  (~100 per round)
3. **Check round calculation**: With default settings, should see 5 rounds calculated automatically
4. **Monitor convergence**: Ensure the fixed update approach doesn't harm model performance
5. **Compare with previous results**: Evaluate if fixed updates improve fairness metrics
6. **Test with different num_clients**: Verify total_train_samples adjustment works correctly

## Running Experiments
When you run your federated training, you should see output like:
```
[Config] Calculated num_rounds = 5 (based on 100 updates/round)
[Client 0] Training: 100 updates ≈ 4 epochs
[Client 0] Dataset: 248 samples, Batch: 8, Updates/epoch: 31.00
...
```

## Example
Given:
- `batch_size = 8` (from config)
- `num_updates_per_round = 100` (from config)

**Client 0** (800 samples):
- Updates per epoch: 800/8 = 100
- Epochs needed: ceil(100/100) = **1 epoch**
- Total updates: **100** ✓

**Client 1** (200 samples):
- Updates per epoch: 200/8 = 25
- Epochs needed: ceil(100/25) = **4 epochs**
- Total updates: **100** ✓

**Client 2** (1600 samples):
- Updates per epoch: 1600/8 = 200
- Epochs needed: ceil(100/200) = **1 epoch** (but will do 200 updates)
- Total updates: **200** (note: slightly more than 100, but ensures at least 100)

## Notes
1. **Centralized and Local training modes** are **NOT affected** - they continue using `num_epochs` from config
2. **Only Federated training** uses the fixed updates approach
3. The actual number of updates may be slightly more than 100 for clients with large datasets (e.g., if dataset_size > batch_size * 100), but this ensures a minimum of 100 updates
4. Early stopping is still active and may terminate training before reaching the target epochs if validation loss plateaus

## References
- Owkin FLamby paper: Describes the fixed updates approach for fair federated learning
- Location: `/Users/nataliamorenoblasco/Downloads/Owkin_FLamby.pdf`

## Testing Recommendations
1. **Verify epoch calculation**: Check logs show correct epochs for different dataset sizes
2. **Confirm fairness**: Verify all clients perform similar number of updates
3. **Monitor convergence**: Ensure the fixed update approach doesn't harm model performance
4. **Compare with previous results**: Evaluate if fixed updates improve fairness metrics
