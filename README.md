# Survival Models in Federated Healthcare Settings

This repository contains the code developed for my thesis on survival analysis in healthcare settings under three training paradigms:

- `federated` learning
- `centralized` learning
- `local` learning

The project uses the `Fed-TCGA-BRCA` dataset from `FLamby` and compares three survival modeling approaches:

- `CoxPH`
- `DeepSurv`
- `RSF_FedSurF`

The main entry point is [`src/main.py`](src/main.py), where the experiment configuration is defined. From that script, the user can choose the training mode, the model, the federated strategy, the participating centers, the number of epochs, and the rest of the experiment hyperparameters.

## Repository structure

```text
.
├── README.md                                      # Project overview and usage guide.
├── requirements.txt                               # Full environment dependencies used during development.
├── clean_requirements.txt                         # Lighter dependency list for cleaner environment recreation.
├── run_job.sh                                     # SLURM job script to launch the experiment on CSC/HPC.
│
├── src/                                           # Main source code for training, evaluation, and experiment control.
│   ├── main.py                                    # Main entry point; choose model, mode, centers, strategy, epochs, and start the run.
│   ├── config.py                                  # Central experiment configuration dataclass and default hyperparameters.
│   ├── dataset_manager.py                         # Loads FLamby data, prepares train/val/test splits, caching, and per-mode datasets.
│   ├── model_manager.py                           # Instantiates the selected survival model from the config.
│   ├── utils.py                                   # Shared evaluation utilities and survival metrics computation.
│   │
│   ├── Models/                                    # Implementations/wrappers of the supported survival models.
│   │   ├── CoxPH.py                               # Cox proportional hazards model implementation used in the thesis.
│   │   ├── DeepSurv.py                            # DeepSurv neural survival model implementation.
│   │   └── RSF_FedSurF.py                         # Random Survival Forest and FedSurF/FedSurF++ related model logic.
│   │
│   ├── Training_Modes/                            # Training pipelines grouped by learning paradigm.
│   │   ├── Centralized_Learning/
│   │   │   └── centralized_run.py                 # Centralized training and evaluation workflow.
│   │   ├── Local_Learning/
│   │   │   └── local_run.py                       # Single-center local training and evaluation workflow.
│   │   └── Federated_Learning/
│   │       ├── strategies.py                      # Flower strategies: FedAvg, FedProx, FedAdam, FedSurvForest, FedSurFPlusPlus.
│   │       ├── clientCoxPH.py                     # Flower client implementation for federated CoxPH.
│   │       ├── clientDeepSurv.py                  # Flower client implementation for federated DeepSurv.
│   │       ├── clientRSFFedSurF.py                # Flower client implementation for federated RSF/FedSurF methods.
│   │       ├── server.py                          # Older/custom server-side strategy prototype for federated Cox aggregation.
│   │       └── task.py                            # Helper utilities for extracting/setting model weights.
│   │
│   └── results/                                   # Runtime output folder created automatically per experiment (not tracked).
│
├── Exps_runs_randomness/                          # Scripts for aggregating repeated-run and randomness experiments.
│   ├── aggregate_randomness_results_RSF.py        # Aggregates repeated-run results for RSF/FedSurF experiments.
│   ├── aggregate_randomness_results_DeepSurv.py   # Aggregates repeated-run results for DeepSurv experiments.
│   └── centralized_aggregate_results_DeepSurv.py  # Aggregates centralized DeepSurv repeated-run results.
│
├── DataExploration/                               # Exploratory analysis material for understanding the dataset.
│   └── dataset_exploration.ipynb                  # Notebook for inspecting the dataset and center distributions.
│
├── ResultsAnalysis/                               # Final analysis figures and plotting scripts grouped by model.
│   ├── CoxPH/
│   │   ├── plot_main_result.py                    # Main CoxPH result plotting script.
│   │   ├── plot_client_level_effect.py            # CoxPH analysis of client-level effects.
│   │   ├── plot_effect_number_of_clients.py       # CoxPH analysis of the number of participating clients.
│   │   ├── figure_1_CoxPH.png                     # Main CoxPH thesis figure 1.
│   │   ├── figure_2_CoxPH.png                     # Main CoxPH thesis figure 2.
│   │   └── figure_3_effect_number_of_clients.png  # CoxPH figure for client-count effect.
│   │
│   ├── DeepSurv/
│   │   ├── plot_main_result.py                    # Main DeepSurv result plotting script.
│   │   ├── plot_client_level_effect.py            # DeepSurv analysis of client-level effects.
│   │   ├── plot_effect_number_of_clients.py       # DeepSurv analysis of the number of participating clients.
│   │   ├── figure_1_DeepSurv.png                  # Main DeepSurv thesis figure 1.
│   │   ├── figure_2_DeepSurv.png                  # Main DeepSurv thesis figure 2.
│   │   └── figure_3_DeepSurv.png                  # Main DeepSurv thesis figure 3.
│   │
│   └── RSF/
│       ├── plot_e1_local_trees.py                 # RSF/FedSurF analysis for local-tree behavior.
│       ├── plot_e2_main_result.py                 # Main RSF/FedSurF result plotting script.
│       ├── plot_e2_client_level_effect.py         # RSF/FedSurF analysis of client-level effects.
│       ├── figure_1_E1_RSF.png                    # RSF thesis figure 1.
│       ├── figure_2_E2_RSF.png                    # RSF thesis figure 2.
│       └── figure_3_E2_RSF.png                    # RSF thesis figure 3.
│
│
└── FLamby/                                        # Local FLamby directory/submodule area if present in the environment.
```

## How the project is controlled from `src/main.py`

[`src/main.py`](src/main.py) is the script that orchestrates the full experiment. The user defines a `Config(...)` object at the bottom of the file and then launches the run.

From this script, the user can select:

- the `training_mode`: `federated`, `centralized`, or `local`
- the `model`: `CoxPH`, `DeepSurv`, or `RSF_FedSurF`
- the federated `strategy`
- the participating `centers`
- the number of `num_epochs`
- the number of local updates per round with `num_updates_per_round`
- the batch size with `batch_size`
- the number of trees for RSF/FedSurF with `n_trees_local` and `n_trees_federated`
- the evaluation grid mode with `eval_grid_mode`
- the optimizer/model hyperparameters defined in [`src/config.py`](src/config.py)

Important details:

- `centers=[...]` determines which healthcare centers participate in the experiment.
- `num_clients` is automatically synchronized with the number of selected centers inside `Config.__post_init__`, so the real number of clients is driven by the `centers` list.
- in `local` mode, the code requires exactly one center
- in `federated` mode, Flower simulations are launched and the strategy is selected automatically depending on the model and `strategy`
- in `centralized` mode, all selected centers are pooled into one global dataset

## Supported training choices

### Training modes

- `federated`: clients train separately and communicate through Flower strategies
- `centralized`: all selected centers are pooled into a single training dataset
- `local`: a single center is trained independently

### Models

- `CoxPH`: neural/network-based Cox proportional hazards implementation used in this repository
- `DeepSurv`: deep survival neural network
- `RSF_FedSurF`: random survival forest variant used for FedSurF and FedSurF++

### Federated strategies

For `DeepSurv` and `CoxPH`, the main strategies available from [`src/Training_Modes/Federated_Learning/strategies.py`](src/Training_Modes/Federated_Learning/strategies.py) are:

- `FedAvg`
- `FedProx`
- `FedAdam`

For `RSF_FedSurF`, the repository includes:

- `FedSurFPlusPlus`

## Example configuration pattern

At the bottom of [`src/main.py`](src/main.py), the user can activate one configuration block and run the experiment. Conceptually, the workflow is:

```python
user_config = Config(
    model="DeepSurv",
    centers=[0, 1, 2],
    training_mode="federated",
    num_clients=3,
    num_epochs=30,
    strategy="FedAvg",
)

main(user_config)
```

Typical changes the user may want to make in `main.py` are:

- switch `training_mode` between `federated`, `centralized`, and `local`
- change `model` to compare `CoxPH`, `DeepSurv`, and `RSF_FedSurF`
- choose different `strategy` values for federated experiments
- change `centers=[...]` to vary the number of participating institutions
- increase or decrease `num_epochs`
- adjust tree counts, batch size, learning rate, or evaluation-grid settings

## Running the code

After creating and activating the environment, the experiment can be launched from the repository root with:

```bash
python src/main.py
```



