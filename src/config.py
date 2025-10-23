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
    model: str = "CoxPH"
    
    # Paths:

    # Training parameters for different modes:
    training_mode: str = "federated"  # Options: "federated", "centralized", "local"

    # Federated Learning parameters:
    num_clients: int = 3  # Number of participating centers/clients (max 6) - has to be the same  as the length of centers list
    strategy: str = "FedAvg"
    num_rounds: int = 2
    num_epochs: int = 2
    batch_size: int = 32
    strategy: str = "FedAvg"
    lr: float = 0.001
    
    # Data parameters:
    centers: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])

    # Runtime paths (initialized in __post_init__):
    results_dir: Path = None
    experiment_dir: Path = None

    def __post_init__(self):
        """Ensure the results directory and experiment-specific subfolder are created."""
        results_dir = Path("results")
        experiment_dir = results_dir / self.experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        # Store the paths for later use:
        self.results_dir = results_dir
        self.experiment_dir = experiment_dir



