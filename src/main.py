# ACTIVATE ENVIRONMENT CSC: [nmorenob@r07c51 Masters_thesis]$ source venv/bin/activate
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
#from Training_Modes.Federated_Learning.strategies import get_strategy
from Training_Modes.Federated_Learning.strategies import CustomFedAvg, aggregate_evaluate_metrics

print("DEBUG: reached end of import section successfully")



def init_logging(config):
    # Debugging: Print the experiment directory and ID
    logger = logging.getLogger("main")
    if logger.hasHandlers():
        return logger  # Avoid re-adding handlers

    # Ensure the directory is created
    os.makedirs(config.experiment_dir, exist_ok=True)

    log_filename = os.path.join(config.experiment_dir, f"experiment_{config.experiment_id}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ]
    )
    logger.info(f"Logging initialized for experiment {config.experiment_id}, log file: {log_filename}")
    return logger


def main(config: Config):
    # Initialize logging:
    logger = init_logging(config)
    logging.getLogger().setLevel(logging.INFO)
    logger.info(f"Experiment started: {config.experiment_id}")

    logger.info(f"[Global] Evaluation times set: {config.global_eval_times}")

    import ray
    ray.init(
        _memory=2 * 1024 * 1024 * 1024,          # Limit Ray worker memory to 2GB
        object_store_memory=512 * 1024 * 1024,   # Limit Ray object store to 512MB
        ignore_reinit_error=True,                # Prevent errors on reinit
        include_dashboard=False                  # Optional: disable Ray dashboard
    )

    # Check training mode and proceed accordingly:
    if config.training_mode == "federated": # Federated Learning Simulation
        def client_fn(cid: str):
            init_logging(config)
            cid_int = int(cid)

            # Load cached data
            dm = DatasetManager(config=config, client_idx=cid_int)
            dataloaders = dm.get_federated_dataloaders()

            # Build model
            model_manager = ModelManager(config)
            model_manager.initialize_model()
            model = model_manager.get_model()

            # Return client
            return FederatedCoxClient(
                cid=cid_int,
                name=config.centers[cid_int],
                model=model,
                config=config,
                dataset_manager=None,
                dataloaders=dataloaders
            )#.to_client()

        

        # Create a list of client instances
        clients = [
            client_fn(str(cid))
            for cid in range(config.num_clients)
        ]

        # The `client_fn` function returns a client instance by CID
        def get_client(cid: str) -> fl.client.Client:
            return clients[int(cid)]

        strategy = CustomFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=config.num_clients,
            min_evaluate_clients=config.num_clients,
            min_available_clients=config.num_clients,
        )

        #strategy = get_strategy(config.strategy)
        logger.info("Starting Federated Learning simulation...")
        fl.simulation.start_simulation(
            client_fn=get_client,
            num_clients=config.num_clients, 
            config=fl.server.ServerConfig(num_rounds=config.num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1},  # limit each client to 1 CPU
        )

    elif config.training_mode == "centralized":
        # Centralized Training
        logger.info("Centralized training mode selected. Exiting.")
        quit()

    elif config.training_mode == "local":
        # Local Training
        logger.info("Local training mode selected. Exiting.")
        quit()


if __name__ == "__main__":
    # Define user-specific configuration
    user_config = Config(
        model="SLR",
        centers=[0, 1, 2, 3, 4, 5],
        training_mode="federated",
        num_clients=6,
        strategy="FedAvg",
        num_rounds=150,
    )
    main(user_config)




#----------------------------------------------
# print("DEBUG: main.py started execution")

# import os, sys

# # ==========================================================
# # [CSC FIX] Configure temporary directories and Ray limits
# # ==========================================================
# # Use a short tmp path on macOS (Darwin), otherwise use CSC scratch
# if os.uname().sysname == "Darwin":
#     scratch_dir = "/tmp/ray_short"  # Short path to avoid AF_UNIX limit
# else:
#     scratch_dir = os.environ.get("TMPDIR", f"/scratch/{os.environ.get('USER', 'default')}/tmp")

# os.makedirs(scratch_dir, exist_ok=True)

# # Tell Python and Ray to use this directory
# os.environ["TMPDIR"] = scratch_dir
# os.environ["RAY_TMPDIR"] = scratch_dir

# # Optional: results folder inside scratch (or tmp on mac)
# RESULTS_DIR = os.path.join(scratch_dir, "results")
# os.makedirs(RESULTS_DIR, exist_ok=True)

# print(f"DEBUG: Using scratch/tmp directory → {scratch_dir}")
# print(f"DEBUG: Results directory → {RESULTS_DIR}")
# # ==========================================================

# print("DEBUG: __name__ =", __name__)
# print("DEBUG: Running file =", os.path.abspath(__file__))
# print("DEBUG: Current working dir =", os.getcwd())

# print("DEBUG: testing imports...")

# import logging
# from config import Config
# from Training_Modes.Federated_Learning.client import FederatedCoxClient
# import flwr as fl
# from dataset_manager import DatasetManager
# from model_manager import ModelManager
# from Training_Modes.Federated_Learning.strategies import CustomFedAvg
# import ray   # [CSC FIX] Added for ray.init

# print("DEBUG: reached end of import section successfully")

# # ==========================================================
# # [CSC FIX] Initialize Ray with safer object store limits
# # ==========================================================
# try:
#     ray.init(
#         object_store_memory=1 * 1024 * 1024 * 1024,  # 1 GB object store
#         ignore_reinit_error=True,
#         _system_config={"automatic_object_spilling_enabled": True}  # spill to disk if full
#     )
#     print("DEBUG: Ray initialized with limited object store (1 GB)")
# except Exception as e:
#     print(f"WARNING: Ray initialization failed: {e}")
# # ==========================================================


# def init_logging(config):
#     logger = logging.getLogger("main")
#     if logger.hasHandlers():
#         return logger  # Avoid re-adding handlers

#     # Ensure the directory is created inside RESULTS_DIR
#     base_dir = RESULTS_DIR if os.path.exists(RESULTS_DIR) else os.path.join(os.getcwd(), "results")
#     experiment_dir = os.path.join(base_dir, config.experiment_id)
#     os.makedirs(experiment_dir, exist_ok=True)
#     log_filename = os.path.join(experiment_dir, f"experiment_{config.experiment_id}.log")

#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
#         handlers=[
#             logging.StreamHandler(),
#             logging.FileHandler(log_filename)
#         ]
#     )
#     logger.info(f"Logging initialized for experiment {config.experiment_id}, log file: {log_filename}")
#     return logger


# def main(config: Config):
#     logger = init_logging(config)
#     logging.getLogger().setLevel(logging.INFO)
#     logger.info(f"Experiment started: {config.experiment_id}")
#     logger.info(f"[Global] Evaluation times set: {config.global_eval_times}")

#     if config.training_mode == "federated":  # Federated Learning Simulation
#         def client_fn(cid: str):
#             init_logging(config)
#             cid_int = int(cid)
#             dm = DatasetManager(config=config, client_idx=cid_int)
#             dataloaders = dm.get_federated_dataloaders()
#             model_manager = ModelManager(config)
#             model_manager.initialize_model()
#             model = model_manager.get_model()
#             return FederatedCoxClient(
#                 cid=cid_int,
#                 name=config.centers[cid_int],
#                 model=model,
#                 config=config,
#                 dataset_manager=None,
#                 dataloaders=dataloaders
#             )

#         clients = [client_fn(str(cid)) for cid in range(config.num_clients)]

#         def get_client(cid: str) -> fl.client.Client:
#             return clients[int(cid)]

#         strategy = CustomFedAvg(
#             fraction_fit=1.0,
#             fraction_evaluate=1.0,
#             min_fit_clients=config.num_clients,
#             min_evaluate_clients=config.num_clients,
#             min_available_clients=config.num_clients,
#         )

#         logger.info("Starting Federated Learning simulation...")
#         fl.simulation.start_simulation(
#             client_fn=get_client,
#             num_clients=config.num_clients,
#             config=fl.server.ServerConfig(num_rounds=config.num_rounds),
#             strategy=strategy,
#         )

#     elif config.training_mode == "centralized":
#         logger.info("Centralized training mode selected. Exiting.")
#         quit()

#     elif config.training_mode == "local":
#         logger.info("Local training mode selected. Exiting.")
#         quit()


# if __name__ == "__main__":
#     user_config = Config(
#         model="SLR",
#         centers=[0, 1, 2],
#         training_mode="federated",
#         num_clients=3,
#         strategy="FedAvg",
#         num_rounds=3,
#         num_epochs=2,
#         batch_size=32,
#     )
#     main(user_config)


# #ls -R /tmp/ray_short/results/
# # code /tmp/ray_short/results/20251102_085420/experiment_20251102_085420.log