# FedSurF++ Implementation Guide (PAPER-EXACT)

## Overview

This is a **paper-exact** implementation of **FedSurF++** with C-Index based tree sampling, following the algorithm described in the paper.

## Algorithm Description (From Paper)

FedSurF++ is a federated survival analysis algorithm that consists of **three distinct phases**:

### Phase 1: Local Training
Each client k trains a local Random Survival Forest M_k with T_k trees using their local data D_k.

### Phase 2: Tree Assignment (Server-side)
The server determines how many trees T'_k each client should send. Algorithm from paper:

```
Input: T (desired trees in global forest), {N_k} (dataset sizes)
Output: {T'_k} (trees per client)

for iteration = 1 to T:
    Sample client k with probability ∝ N_k
    Increment T'_k
Ensure T'_k ≤ T_k for all clients
```

### Phase 3: Tree Sampling (Client-side)
Each client evaluates their trees with C-Index and samples T'_k trees proportionally:

```
For each client k:
    1. Evaluate each tree j on validation set → C-Index_j
    2. Build probability: p_j = C-Index_j / Σ(C-Index_j)
    3. Sample T'_k trees WITHOUT replacement using p_j
    4. Send selected trees to server
```

## Implementation Flow (Two Communication Rounds)

### **Round 1: Metadata + Tree Assignment**

**Clients:**
1. Train local RSF with T_k trees
2. Send metadata: `{Tk: T_k, Nk: N_k}` to server

**Server:**
1. Collect metadata from all clients
2. Perform tree assignment (iterative sampling with probability ∝ N_k)
3. Compute T'_k for each client
4. Send T'_k back to each client

### **Round 2: Tree Sampling + Transfer**

**Clients:**
1. Receive T'_k from server
2. Evaluate ALL local trees with C-Index on validation set
3. Sample T'_k trees proportional to C-Index scores
4. Send selected T'_k trees to server

**Server:**
1. Collect trees from all clients
2. Concatenate into global forest
3. Broadcast global forest

### **Round 3+: Evaluation**

**Clients:**
1. Receive global forest
2. Evaluate on test set
3. Report metrics (C-Index, AUC, IBS)

## Key Architectural Points

### Why Two Rounds?

1. **Privacy**: Clients don't send ALL trees, only T'_k selected trees
2. **Communication**: Reduces network load - clients only send needed trees
3. **Paper-Exact**: Follows Algorithm 1 exactly as written

### Tree Assignment (Phase 2)

The server does **NOT** do the tree sampling. It only:
- Determines how many trees each client contributes
- Uses dataset size as proxy for representativeness

### Tree Sampling (Phase 3)

**Clients** do the sampling (NOT server):
- Each client evaluates their own trees locally
- Sampling happens client-side to preserve privacy
- Only send selected trees to server

## Implementation Components

### 1. Model: `Models/RSF_FedSurF.py`

**Class:** `RSFFedSurFPlus`

Standard RSF wrapper with FedSurF-specific functionality.

### 2. Client: `Training_Modes/Federated_Learning/clientRSFFedSurF.py`

**Class:** `FederatedRSFFedSurFClient`

**fit() handles TWO different rounds:**

```python
def fit(self, ins):
    round_type = ins.config.get("round_type")
    
    if round_type == "metadata":
        return self._fit_round1_metadata()  # Send Tk, Nk
    elif round_type == "tree_sampling":
        return self._fit_round2_sampling(ins)  # Sample and send trees
```

**Round 1:**
- Train RSF
- Return: `{Tk, Nk}`

**Round 2:**
- Receive T'_k
- Evaluate trees with C-Index
- Sample T'_k trees
- Return: selected trees

### 3. Strategy: `Training_Modes/Federated_Learning/strategies.py`

**Class:** `FedSurFPlusPlus`

**aggregate_fit() routes by round:**

```python
def aggregate_fit(self, server_round, results, failures):
    if server_round == 1:
        return self._aggregate_round1_metadata(results)
    elif server_round == 2:
        return self._aggregate_round2_trees(results)
```

**Round 1:**
1. Collect metadata
2. Perform tree assignment
3. Store T'_k for each client
4. Return None

**Round 2:**
1. Collect trees from clients
2. Build global forest
3. Return global forest

**configure_fit() configures each round:**

```python
def configure_fit(self, server_round, ...):
    if server_round == 1:
        return [(client, FitIns(config={"round_type": "metadata"}))]
    elif server_round == 2:
        return [(client, FitIns(config={"round_type": "tree_sampling", 
                                         "Tk_prime": T'_k}))]
```

### 4. Configuration: `config.py`

FedSurF++ automatically uses **3 rounds**:
- Round 1: Metadata
- Round 2: Tree sampling
- Round 3: Evaluation

```python
if self.model == "RSF_FedSurF":
    self.num_rounds = 3
```

## Usage

```python
from src.config import Config
from src.main import main

user_config = Config(
    model="RSF_FedSurF",
    centers=[0, 1, 2, 3, 4],
    training_mode="federated",
    num_clients=5,
    strategy="FedSurFPlusPlus",
    n_trees_local=200,        # Each client trains 200 trees
    n_trees_federated=80,     # Global forest has 80 trees
    eval_grid_mode="global"
)

main(user_config)
```

## Communication Pattern

```
ROUND 1:
Client 0 → Server: {Tk: 200, Nk: 120}
Client 1 → Server: {Tk: 200, Nk: 110}
...
Server computes: T'_0=15, T'_1=18, ...
Server → Client 0: T'_k=15
Server → Client 1: T'_k=18
...

ROUND 2:
Client 0: Evaluates 200 trees, samples 15 best
Client 0 → Server: [15 trees]
Client 1: Evaluates 200 trees, samples 18 best
Client 1 → Server: [18 trees]
...
Server: Concatenates all trees → Global forest (80 trees)
Server → All: Global forest

ROUND 3:
Server → All: Global forest
All → Server: Evaluation metrics
```

## Key Differences from Original Implementation

| Aspect | FedSurvForest | FedSurF++ (PAPER-EXACT) |
|--------|---------------|-------------------------|
| Communication Rounds | 1 | 2 |
| Tree Assignment | Implicit | Explicit (Algorithm 1, lines 4-7) |
| Who Does Sampling | Server | **Clients** |
| Privacy | Sends all trees | Sends only T'_k selected trees |
| Follows Paper | Partial | **Complete** |

## Expected Output

```
[FedSurF++] Initialized with T=80
[FedSurF++][Round 1] Metadata Collection + Tree Assignment
[FedSurF++][Round 1] Client 0: Tk=200, Nk=120
[FedSurF++][Round 1] Client 1: Tk=200, Nk=110
...
[FedSurF++][Round 1] Tree Assignment Result: Client 0: T'_k=15 | Client 1: T'_k=18 | ...

[FedSurF++][Round 2] Tree Collection + Global Forest
[Client 0][Round 2] Tree Assignment: T'_k=15
[Client 0][Round 2] Evaluating trees with C-Index...
[Client 0][Round 2] C-Index: mean=0.6234, range=[0.58, 0.67]
[Client 0][Round 2] Sampled 15 trees: C-Index range=[0.61, 0.67]
[FedSurF++][Round 2] Client 0: received 15 trees
[FedSurF++][Round 2] Global forest built: 80 trees (target: 80)

[FedSurF++][Round 3] Evaluation
[Client 0] Test Metrics → C-index=0.6543, AUC=0.6821, IBS=0.1532
```

## Troubleshooting

**Issue:** "Cannot exceed available trees"
- **Cause:** T'_k > T_k for some client
- **Solution:** Config automatically caps T'_k ≤ T_k

**Issue:** "Received fewer trees than target"
- **Cause:** Clients don't have enough trees total
- **Solution:** Increase `n_trees_local`

**Issue:** "round_type not found"
- **Cause:** Using old client with new strategy
- **Solution:** Ensure using `FederatedRSFFedSurFClient`

## Summary

This implementation is **paper-exact**:
- ✅ Two-phase communication (metadata → tree sampling)
- ✅ Server does tree assignment (Algorithm 1, lines 4-7)
- ✅ **Clients** do tree sampling (Algorithm 1, lines 8-10)
- ✅ C-Index based sampling (local evaluation)
- ✅ Privacy-preserving (only send selected trees)

All components follow the paper's Algorithm 1 precisely.
