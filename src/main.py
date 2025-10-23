print("DEBUG: main.py started execution")

import os, sys
print("DEBUG: __name__ =", __name__)
print("DEBUG: Running file =", os.path.abspath(__file__))
print("DEBUG: Current working dir =", os.getcwd())

print("DEBUG: testing imports...")

import logging
from config import Config
from Training_Modes.Federated_Learning.client import FederatedCoxClient
import flwr as fl
from dataset_manager import DatasetManager
from model_manager import ModelManager
from Training_Modes.Federated_Learning.strategies import get_strategy

print("DEBUG: reached end of import section successfully")


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
    logging.getLogger().setLevel(logging.INFO)
    logger.info(f"Experiment started: {config.experiment_id}")
    




    # Check training mode and proceed accordingly:
    if config.training_mode == "federated": # Federated Learning Simulation

        
        # def client_fn(cid: str):
        #     cid_int = int(cid)
        #     client_name = config.centers[cid_int]
            
        #     dm = DatasetManager(config=config, client_idx=cid_int)
        #     model = ModelManager(config).get_model()

        #     return FederatedCoxClient(
        #         cid=cid_int,
        #         name=client_name,
        #         model=model,
        #         config=config,
        #         dataset_manager=dm
        #     )

        def client_fn(cid: str):
            cid_int = int(cid)
            client_name = config.centers[cid_int]

            # Create the dataset manager for this client
            dm = DatasetManager(config=config, client_idx=cid_int)

            # Initialize model correctly
            model_manager = ModelManager(config)
            model_manager.initialize_model()
            model = model_manager.get_model()

            # Create the Flower client
            return FederatedCoxClient(
                cid=cid_int,
                name=client_name,
                model=model,
                config=config,
                dataset_manager=dm
            )
        strategy = get_strategy(config.strategy)
        logger.info("Starting Federated Learning simulation...")
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=config.num_clients, 
            config=fl.server.ServerConfig(num_rounds=config.num_rounds),
            strategy=strategy,
        )

    elif config.training_mode == "centralized":
        # Centralized Training
        quit()
 

    elif config.training_mode == "local":
        # Local Training
        quit()






if __name__ == "__main__":
    # Define user-specific configuration
    user_config = Config(
        model="CoxPH",
        centers=[0, 1, 2],
        training_mode="federated",
        num_clients=3,
        strategy="FedAvg",
        num_rounds=2,
        num_epochs=2,
        batch_size=32,
    )
    print(f"DEBUG: Config created at {user_config.experiment_dir}")

    import traceback

    try:
        main(user_config)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
