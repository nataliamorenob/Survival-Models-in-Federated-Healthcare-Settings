from datetime import datetime
from dataclasses import dataclass, field
import os
from pathlib import Path



def get_current_timestamp() -> str:
    """Returns the current timestamp as a formatted string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class Config:
    # Hyperparameters and settings for the experiment:
    experiment_id: str = field(default_factory=get_current_timestamp)
    model: str = "CoxPH" # Options: "CoxPH", "SLR"
    
    # Paths:

    # Training parameters for different modes:
    training_mode: str = "federated"  # Options: "federated", "centralized", "local"

    # Federated Learning parameters:
    num_clients: int = 3  # Number of participating centers/clients (max 6) - has to be the same  as the length of centers list
    strategy: str = "FedAvg"  # Options: "FedAvg", "FedProx", "FedAdam", "FedSurvForest", etc. Used for DeepSurv and RSF federated strategies.
    num_rounds: int = 2
    num_epochs: int = 2
    batch_size: int = 32  # Larger batches for stability (was 16)
    num_time_bins: int = 100
    strategy: str = "FedAvg"
    lr: float = 0.0005  # Reduced from 0.001 for more stable training
    
    # Data parameters:
    centers: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])

    # Runtime paths (initialized in __post_init__):
    results_dir: Path = None
    experiment_dir: Path = None
    cache_dir: Path = None 

    # Metrics specific parameters
    #global_eval_times: list = field(default_factory=lambda: [100, 489, 878, 1267, 1656, 2044, 2433, 2822, 3211, 3500])
    eval_times_per_client: dict = field(default_factory=dict) # 
    eval_grid_mode: str = "client"  # "client" or "global"
    global_eval_times: list = None
    union_time_grid: list = None



    # RANDOM SURVIVAL FOREST ALGORITHM PARAMETERS (DEFAULT FOR NOW, BUT TUNE IN THE FUTURE):
    n_trees_local: int = 100 # number of trees each client trains (in the future hyperparameter tuning)
    n_trees_federated: int = 200 # number of trees the server samples (in the future hyperparameter tuning)
    n_trees_federated: int = 200 # number of trees the server samples (in the future hyperparameter tuning)
    min_samples_split: int = 5
    min_samples_leaf: int = 10
    #random_state: int = 42

    # DEEPSURV PARAMETERS (optimized for stability):
    deepsurv_hidden_layers: list = field(default_factory=lambda: [64, 32])  # Balanced architecture #64,32
    deepsurv_dropout: float = 0.2  # Increased from 0.1 for better generalization
    deepsurv_batch_norm: bool = False  # Keep disabled for small batches
    deepsurv_activation: str = 'ReLU'
    deepsurv_l2_reg: float = 0.001  # Increased from 0.0001 for stronger regularization




    def __post_init__(self):
        """Ensure the results directory and experiment-specific subfolder are created."""
        # Locate project root (parent of src/)
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "src" / "results"

        # Creation of unique experiment folder
        experiment_dir = results_dir / self.experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Creation of a cache directory for preprocessed datasets
        cache_dir = experiment_dir / "cached_datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save paths to object
        self.results_dir = results_dir
        self.experiment_dir = experiment_dir
        self.cache_dir = cache_dir

# Global feature list for the TCGA-BRCA dataset
ALL_FEATURE_COLUMNS = [f"feature_{i}" for i in range(39)]



