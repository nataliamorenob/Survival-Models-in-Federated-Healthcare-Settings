class Config:
        model: str = "CoxPH"
        centers: list = ['0', '1', '2', '3', '4', '5', '6']
        training_mode: str = "federated" # Options: "federated", "centralized", "local"
        num_clients: int = 3 # Number of participating centers/clients (max 6)
        strategy: str = "FedAvg"
        num_rounds: int = 2
        num_epochs: int = 2
        batch_size: int = 32
