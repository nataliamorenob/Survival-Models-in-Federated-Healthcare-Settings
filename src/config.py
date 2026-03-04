from datetime import datetime
from dataclasses import dataclass, field
import os
from pathlib import Path



def get_current_timestamp() -> str:
    """Returns the current timestamp as a formatted string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_nb_max_rounds(num_updates, batch_size, num_clients=5, num_epochs_pooled=30, total_train_samples=578):
    """
    Calculate the number of federated rounds needed to match the computational budget
    of centralized training.
    
    Formula from FLamby:
    num_rounds = (total_samples // num_clients // batch_size) * num_epochs_pooled // num_updates_per_round
    
    This ensures federated training does the same total number of gradient updates
    as centralized training would do in num_epochs_pooled epochs.
    
    Args:
        num_updates: Number of local gradient updates per client per round
        batch_size: Batch size used for training
        num_clients: Number of federated clients (default: 5 for TCGA-BRCA centers 0-4)
        num_epochs_pooled: Target epochs for centralized training (default: 30)
        total_train_samples: Actual training samples (default: 578 for 5 centers)
    
    Returns:
        Number of federated rounds needed
    
    Example:
        With 578 samples, 5 clients, batch_size=8, num_epochs_pooled=30, num_updates=100:
        - Avg samples per client: 578 // 5 = 115
        - Updates per epoch: 115 // 8 = 14
        - Total updates for 30 epochs: 14 * 30 = 420
        - Rounds needed: 420 // 100 = 4
    """
    avg_samples_per_client = total_train_samples // num_clients
    updates_per_epoch = avg_samples_per_client // batch_size
    total_updates_centralized = updates_per_epoch * num_epochs_pooled
    num_rounds = total_updates_centralized // num_updates
    
    return max(1, num_rounds)  # Ensure at least 1 round


@dataclass
class Config:
    # Hyperparameters and settings for the experiment:
    experiment_id: str = field(default_factory=get_current_timestamp)
    model: str = "CoxPH" # Options: "CoxPH", "SLR"
    
    # Paths:

    # Training parameters for different modes:
    training_mode: str = "federated"  # Options: "federated", "centralized", "local"

    # Federated Learning parameters:
    num_clients: int = 5  # Number of participating centers/clients (max 6) - has to be the same  as the length of centers list
    strategy: str = "FedAvg"  # Options: "FedAvg", "FedProx", "FedAdam", "FedSurvForest", etc. Used for DeepSurv and RSF federated strategies.
    num_rounds: int = None  # Will be calculated dynamically via get_nb_max_rounds() in __post_init__
    num_epochs: int = 30  # Used for centralized/local training AND as num_epochs_pooled for round calculation
    num_updates_per_round: int = 100  # Fixed number of gradient updates per client per round (Owkin FLamby approach)
    #batch_size: int = 32  # Larger batches for stability (was 16)
    batch_size: int = 8
    total_train_samples: int = 578  # Actual training samples for 5 centers (0-4) - validation done separately
    #num_time_bins: int = 100
    strategy: str = "FedAvg"
    #lr: float = 0.0005  # Reduced from 0.001 for more stable training
    lr: float = 0.1
    
    # Data parameters:
    centers: list = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Using 5 centers (excluding center 5)

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
        
        # Calculate num_rounds dynamically if not set (for federated training)
        if self.num_rounds is None and self.training_mode == "federated":
            self.num_rounds = get_nb_max_rounds(
                num_updates=self.num_updates_per_round,
                batch_size=self.batch_size,
                num_clients=self.num_clients,
                num_epochs_pooled=self.num_epochs,
                total_train_samples=self.total_train_samples
            )
            print(f"[Config] Calculated num_rounds = {self.num_rounds} (based on {self.num_updates_per_round} updates/round)")
        elif self.num_rounds is None:
            # For centralized/local training, num_rounds is not used
            self.num_rounds = 1

# Global feature list for the TCGA-BRCA dataset
ALL_FEATURE_COLUMNS = [f"feature_{i}" for i in range(39)]



