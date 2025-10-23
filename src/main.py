import logging
import os
from config import Config
from Training_Modes.Federated_Learning.client import FederatedCoxClient
from config import Config
import flwr as fl
from dataset_manager import DatasetManager
from model_manager import ModelManager
from Training_Modes.Federated_Learning.strategies import get_strategy

# Configure logging for experiments:
def init_logging(config):
    # Debugging: Print the experiment directory and ID
    print(f"Debug: experiment_id = {config.experiment_id}")
    print(f"Debug: experiment_dir = {config.experiment_dir}")

    # Log the experiment directory path
    logger = logging.getLogger("main")
    logger.info(f"Attempting to create experiment directory at: {config.experiment_dir}")

    # Ensure the directory is created
    os.makedirs(config.experiment_dir, exist_ok=True)
    print(f"Experiment directory created at: {config.experiment_dir}")

    log_filename = os.path.join(config.experiment_dir, f"experiment_{config.experiment_id}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ]
    )
    logger = logging.getLogger("main")
    logger.info(f"Logging initialized for experiment {config.experiment_id}, log file: {log_filename}")
    return logger




def main(config: Config):
    # Initialize logging:
    logger = init_logging(config)
    logger.info(f"Experiment started: {config.experiment_id}")



    # Check training mode and proceed accordingly:
    if config["training_mode"] == "federated": # Federated Learning Simulation

        
        def client_fn(cid: str):
            cid_int = int(cid)
            client_name = config.dataset[cid_int]
            
            dm = DatasetManager(config=config, client_idx=cid_int)
            model = ModelManager(config).get_model()

            return FederatedCoxClient(
                cid=cid_int,
                name=client_name,
                model=model,
                config=config,
                dataset_manager=dm
            )
        strategy=get_strategy(config.strategy, num_clients=config.num_clients, num_rounds=config.num_rounds)
        logger.info("Starting Federated Learning simulation...")
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=config.num_clients, 
            config=fl.server.ServerConfig(num_rounds=config.num_rounds),
            strategy=strategy,
        )

    elif config["training_mode"] == "centralized":
        # Centralized Training
        quit()
 

    elif config["training_mode"] == "local":
        # Local Training
        quit()






if __name__ == "__main__":
    # Define user-specific configuration
    user_config = Config(
        model="CoxPH",
        centers=['0', '1', '2'],
        training_mode=True,
        num_clients=3,
        strategy="FedAvg",
        num_rounds=2,
        num_epochs=2,
        batch_size=32,
    )

    print("Debug: Script has entered the __name__ == '__main__' block.")

    try:
        # Call main function with user configuration
        main(user_config)
    except Exception as e:
        print(f"An error occurred: {e}")