# # ACTIVATE ENVIRONMENT CSC: [nmorenob@r07c51 Masters_thesis]$ source venv/bin/activate
# print("DEBUG: main.py started execution")

# import os, sys
# print("DEBUG: __name__ =", __name__)
# print("DEBUG: Running file =", os.path.abspath(__file__))
# print("DEBUG: Current working dir =", os.getcwd())

# print("DEBUG: testing imports...")

# import logging
# from config import Config
# from Training_Modes.Federated_Learning.client import FederatedCoxClient
# import flwr as fl
# import numpy as np
# from dataset_manager import DatasetManager
# from model_manager import ModelManager
# #from Training_Modes.Federated_Learning.strategies import get_strategy
# from Training_Modes.Federated_Learning.strategies import CustomFedAvg, FedSurvForest, aggregate_evaluate_metrics

# print("DEBUG: reached end of import section successfully")



# def init_logging(config):
#     # Debugging: Print the experiment directory and ID
#     logger = logging.getLogger("main")
#     if logger.hasHandlers():
#         return logger  # Avoid re-adding handlers

#     # Ensure the directory is created
#     os.makedirs(config.experiment_dir, exist_ok=True)

#     log_filename = os.path.join(config.experiment_dir, f"experiment_{config.experiment_id}.log")

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
#     # Initialize logging:
#     logger = init_logging(config)
#     logging.getLogger().setLevel(logging.INFO)
#     logger.info(f"Experiment started: {config.experiment_id}")

#     #logger.info(f"[Global] Evaluation times set: {config.global_eval_times}")

#     import ray
#     ray.init(
#         _memory=2 * 1024 * 1024 * 1024,          # Limit Ray worker memory to 2GB
#         object_store_memory=512 * 1024 * 1024,   # Limit Ray object store to 512MB
#         ignore_reinit_error=True,                # Prevent errors on reinit
#         include_dashboard=False                  # Optional: disable Ray dashboard
#     )

#     # Check training mode and proceed accordingly:
#     if config.training_mode == "federated": # Federated Learning Simulation
#         def client_fn(cid: str):
#             init_logging(config)
#             cid_int = int(cid)

#             # Load cached data
#             dm = DatasetManager(config=config, client_idx=cid_int)
#             dataloaders = dm.get_federated_dataloaders()

#             # Build model
#             model_manager = ModelManager(config)
#             model_manager.initialize_model()
#             model = model_manager.get_model()

#             if config.model == "RSF":
#                 from Training_Modes.Federated_Learning.clientFedSurF import FederatedRSFClient
#                 return FederatedRSFClient(
#                     cid=cid_int,
#                     name=config.centers[cid_int],
#                     model=model,
#                     config=config,
#                     dataloaders=dataloaders
#                     # there is no need to pass the model bc FEderatedRSFClient
#                 ).to_client()
#             elif config.model == "SLR":
#                 # Return client
#                 return FederatedCoxClient(
#                     cid=cid_int,
#                     name=config.centers[cid_int],
#                     model=model,
#                     config=config,
#                     dataset_manager=None,
#                     dataloaders=dataloaders
#                 )#.to_client()


#         # NEW --> GLOBAL EVAL TIME GRID
#         if config.eval_grid_mode == "global":
#             all_times = []
#             for cid in range(config.num_clients):
#                 all_times.extend(config.eval_times_per_client[cid])

#             all_times = np.array(all_times)

#             # Option B: quantile-compress (recommended)
#             union_grid = np.unique(all_times)
#             global_grid = np.quantile(union_grid, np.linspace(0.05, 0.95, 100))

#             config.global_eval_times = global_grid.tolist()

#             print(f"\n[GLOBAL] Created global evaluation grid ({len(global_grid)} points)")
#             print(global_grid)

#         # Create a list of client instances
#         clients = [
#             client_fn(str(cid))
#             for cid in range(config.num_clients)
#         ]

#         # The `client_fn` function returns a client instance by CID
#         def get_client(cid: str) -> fl.client.Client:
#             return clients[int(cid)]


#         # Selection of FL strategy based on config.strategy parameter:
#         if config.strategy == "FedSurvForest":
#             logger.info("[Global] Using FedSurvForest strategy")
#             strategy = FedSurvForest(
#                 fraction_fit=1.0,
#                 fraction_evaluate=1.0,
#                 min_fit_clients=config.num_clients,
#                 min_evaluate_clients=config.num_clients,
#                 min_available_clients=config.num_clients,
#                 # extra hyperparameters needed by FedSurvForest
#                 num_trees_fed=config.n_trees_federated
#             )

#         elif config.strategy == "CustomFedAvg":
#             logger.info("[Global] Using CustomFedAvg strategy")
#             strategy = CustomFedAvg(
#                 fraction_fit=1.0,
#                 fraction_evaluate=1.0,
#                 min_fit_clients=config.num_clients,
#                 min_evaluate_clients=config.num_clients,
#                 min_available_clients=config.num_clients
#             )

#         else:
#             raise ValueError(f"No FL strategy defined.")
        
        
#         # strategy = CustomFedAvg(
#         #     fraction_fit=1.0,
#         #     fraction_evaluate=1.0,
#         #     min_fit_clients=config.num_clients,
#         #     min_evaluate_clients=config.num_clients,
#         #     min_available_clients=config.num_clients,
#         # )

#         #strategy = get_strategy(config.strategy)
#         logger.info("Starting Federated Learning simulation...")
#         fl.simulation.start_simulation(
#             client_fn=get_client,
#             num_clients=config.num_clients, 
#             config=fl.server.ServerConfig(num_rounds=config.num_rounds),
#             strategy=strategy,
#             client_resources={"num_cpus": 1},  # limit each client to 1 CPU
#         )

#     elif config.training_mode == "centralized":
#         # Centralized Training
#         logger.info("Centralized training mode selected. Exiting.")
#         quit()

#     elif config.training_mode == "local":
#         # Local Training
#         logger.info("Local training mode selected. Exiting.")
#         quit()


# if __name__ == "__main__":
#     # Define user-specific configuration
#     user_config = Config(
#         model="RSF",
#         centers=[0, 1, 2, 3],
#         training_mode="federated",
#         num_clients=4,
#         strategy="FedSurvForest",
#         num_rounds=2,
#         eval_grid_mode="global" # or "client"
#     )
#     main(user_config)


# ACTIVATE ENVIRONMENT CSC: [nmorenob@r07c51 Masters_thesis]$ source venv/bin/activate
print("DEBUG: main.py started execution")

import os, sys
from pathlib import Path
import random


# ADD PROJECT ROOT TO PYTHONPATH:
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


print("DEBUG: __name__ =", __name__)
print("DEBUG: Running file =", os.path.abspath(__file__))
print("DEBUG: Current working dir =", os.getcwd())

print("DEBUG: testing imports...")

import logging
import numpy as np
import torch
import flwr as fl
from config import Config
from dataset_manager import DatasetManager
from model_manager import ModelManager
from Training_Modes.Federated_Learning.strategies import (
    CustomFedAvg, FedSurvForest, DeepSurvFedAvg, aggregate_evaluate_metrics
)
from Training_Modes.Federated_Learning.client import FederatedCoxClient
from Training_Modes.Centralized_Learning.centralized_run import run_centralized
from Training_Modes.Local_Learning.local_run import run_local

print("DEBUG: reached end of import section successfully")


def init_logging(config):
    logger = logging.getLogger("main")
    if logger.hasHandlers():
        return logger

    os.makedirs(config.experiment_dir, exist_ok=True)
    log_filename = os.path.join(config.experiment_dir, f"experiment_{config.experiment_id}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_filename)],
    )

    logger.info(f"Logging initialized for experiment {config.experiment_id}, log file: {log_filename}")
    return logger


def main(config: Config):
    # Random seeds to test randomness
    # Using well-tested seeds from ML research (42, 123, 456, 789, 1024, 2048)
    run_id = int(os.environ.get("RUN_ID", 0))
    #STABLE_SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192]
    STABLE_SEEDS = [42, 1337, 123, 456, 789, 1024, 8192, 777, 2026, 9999, 555]
    seed = STABLE_SEEDS[run_id] if run_id < len(STABLE_SEEDS) else 42 + run_id

    config.random_state = seed

    # Set ALL random seeds (Python, NumPy, PyTorch)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU training
    
    # Make PyTorch deterministic (may impact performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Init logging
    logger = init_logging(config)
    logging.getLogger().setLevel(logging.INFO)
    logger.info(f"Experiment started: {config.experiment_id}")

    if config.training_mode == "federated":
        logger.info("Federated training mode selected.")
    elif config.training_mode == "centralized":
        logger.info("Centralized training mode selected.")
        run_centralized(config)
        return
    elif config.training_mode == "local":
        logger.info("Local training mode selected.")
        if config.num_clients != 1:
            raise ValueError(
                f"Local mode requires num_clients=1, got {config.num_clients}"
            )
        run_local(config)
        return
    else:
        raise ValueError(f"Unknown training_mode: {config.training_mode}")

    # Init RAY (federated only) - minimal memory footprint
    import ray
    import gc
    
    # Disable Ray's colored output for cleaner logs
    os.environ["RAY_COLOR_PREFIX"] = "0"
    
    ray.init(
        _memory=1 * 1024 * 1024 * 1024,  # 1GB per worker (reduced to fit 5 clients in 16GB)
        object_store_memory=512 * 1024 * 1024,  # 512MB object store
        ignore_reinit_error=True,
        include_dashboard=False,
        logging_level="ERROR",  # Reduce Ray's verbose logging
    )

    # Build eval_times_per_client by loading metadata only (no full preload)
    print("\n[METADATA] Loading eval_times for all clients...")
    for cid in range(config.num_clients):
        dm = DatasetManager(config=config, client_idx=cid)
        _ = dm.get_federated_dataloaders()  # This populates eval_times_per_client
        
        if cid not in config.eval_times_per_client:
            raise RuntimeError(
                f"[FATAL] DatasetManager did NOT populate eval_times_per_client[{cid}]!"
            )
        
        print(f"[METADATA] Client {cid}: eval_times len={len(config.eval_times_per_client[cid])}")
        
        # Force garbage collection after each client to free memory
        gc.collect()

    print("[METADATA] All eval_times loaded.\n")

    # =====================================================================
    # 2. BUILD GLOBAL GRID (Only after eval_times_per_client exists)
    # =====================================================================
    if config.eval_grid_mode == "global":
        print("[GLOBAL] Building global evaluation grid...")

        all_times = []
        for cid in range(config.num_clients):
            all_times.extend(config.eval_times_per_client[cid])

        all_times = np.array(all_times)
        union_grid = np.unique(all_times)
        config.union_time_grid = union_grid

        global_grid = np.quantile(union_grid, np.linspace(0.05, 0.95, 100))
        config.global_eval_times = global_grid.tolist()

        print(f"[GLOBAL] Created global evaluation grid with {len(global_grid)} points.")
        print(global_grid)
        print()

    # =====================================================================
    # 3. CLIENT FACTORY — Now uses preloaded data
    # =====================================================================
    def client_fn(cid: str):
        init_logging(config)
        cid_int = int(cid)

        # Load data on-demand (not preloaded) to save memory
        dm = DatasetManager(config=config, client_idx=cid_int)
        dataloaders = dm.get_federated_dataloaders()

        model_manager = ModelManager(config, client_id=cid_int)
        model_manager.initialize_model()
        model = model_manager.get_model()

        if config.model == "RSF":
            from Training_Modes.Federated_Learning.clientFedSurF import FederatedRSFClient
            return FederatedRSFClient(
                cid=cid_int,
                name=config.centers[cid_int],
                model=model,
                config=config,
                dataloaders=dataloaders
            ).to_client()

        elif config.model == "RSF_FedSurF":
            from Training_Modes.Federated_Learning.clientRSFFedSurF import FederatedRSFFedSurFClient
            return FederatedRSFFedSurFClient(
                cid=cid_int,
                name=config.centers[cid_int],
                model=model,
                config=config,
                dataloaders=dataloaders
            ).to_client()

        elif config.model == "DeepSurv":
            from Training_Modes.Federated_Learning.clientDeepSurv import FederatedDeepSurvClient
            return FederatedDeepSurvClient(
                cid=cid_int,
                name=config.centers[cid_int],
                model=model,
                config=config,
                dataloaders=dataloaders
            ).to_client()

        elif config.model == "CoxPH":
            from Training_Modes.Federated_Learning.clientCoxPH import FederatedCoxPHClient
            return FederatedCoxPHClient(
                cid=cid_int,
                name=config.centers[cid_int],
                model=model,
                config=config,
                dataloaders=dataloaders
            ).to_client()

        elif config.model == "SLR":
            return FederatedCoxClient(
                cid=cid_int,
                name=config.centers[cid_int],
                model=model,
                config=config,
                dataset_manager=None,
                dataloaders=dataloaders
            )

    # =====================================================================
    # 4. Build Strategy
    # =====================================================================
    if config.model in ("DeepSurv", "CoxPH"):
        # Select FL strategy for neural Cox models based on config.strategy
        from Training_Modes.Federated_Learning.strategies import get_strategy
        from flwr.common import ndarrays_to_parameters
        
        # Create initial model to get initial parameters for the strategy
        logger.info("[Global] Initializing model to get initial parameters for strategy")
        model_manager = ModelManager(config, client_id=0)
        model_manager.initialize_model()
        initial_model = model_manager.get_model()
        
        # Extract initial parameters as numpy arrays
        initial_params = [
            param.cpu().detach().numpy() 
            for param in initial_model.network.state_dict().values()
        ]
        
        # Convert to Flower Parameters format
        initial_parameters = ndarrays_to_parameters(initial_params)
        
        logger.info(f"[Global] Using {config.strategy} strategy for {config.model}")
        strategy = get_strategy(
            config.strategy,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=config.num_clients,
            min_evaluate_clients=config.num_clients,
            min_available_clients=config.num_clients,
            initial_parameters=initial_parameters,
        )
    elif config.strategy == "FedSurvForest":
        logger.info("[Global] Using FedSurvForest strategy")
        strategy = FedSurvForest(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=config.num_clients,
            min_evaluate_clients=config.num_clients,
            min_available_clients=config.num_clients,
            num_trees_fed=config.n_trees_federated
        )

    elif config.strategy == "FedSurFPlusPlus":
        from Training_Modes.Federated_Learning.strategies import FedSurFPlusPlus
        logger.info("[Global] Using FedSurF++ strategy (C-Index based)")
        strategy = FedSurFPlusPlus(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=config.num_clients,
            min_evaluate_clients=config.num_clients,
            min_available_clients=config.num_clients,
            num_trees_fed=config.n_trees_federated
        )

    elif config.strategy == "CustomFedAvg":
        logger.info("[Global] Using CustomFedAvg strategy")
        strategy = CustomFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=config.num_clients,
            min_evaluate_clients=config.num_clients,
            min_available_clients=config.num_clients,
        )
    else:
        raise ValueError(f"No FL strategy defined: {config.strategy}")

    # =====================================================================
    # 5. Start Simulation
    # =====================================================================
    logger.info("Starting Federated Learning simulation...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )


if __name__ == "__main__":
    # FEDERATED DEEPSURV TRAINING (Local Risk Set Approximation):
    # user_config = Config(
    #     model="DeepSurv",
    #     centers=[0, 1, 2, 3, 4],
    #     training_mode="federated",
    #     num_clients=5,
    #     num_epochs=30,  # Local epochs per round
    #     strategy="FedProx"  # FedAdam, FedProx and FedAvg available for DeepSurv
    # )

    # FEDERATED RSF TRAINING (Original FedSurF):
    # user_config = Config(
    #     model="RSF",
    #     centers=[0, 1, 2, 3, 4],
    #     training_mode="federated",
    #     num_clients=5,
    #     strategy="FedSurvForest",
    #     eval_grid_mode="global"  # or "client"
    # )

    # FEDERATED RSF_FedSurF TRAINING (FedSurF++ with C-Index):
    user_config = Config(
        model="RSF_FedSurF",
        centers=[4],
        training_mode="local",
        num_clients=1,
        strategy="FedSurFPlusPlus",
        eval_grid_mode="global"  # or "client"
    )

    # # CENTRALIZED TRAINING:
    # user_config = Config(
    #     model="RSF",
    #     centers=[0, 1, 2],
    #     training_mode="centralized",
    #     num_clients=3,
    #     num_rounds=2,
    #     eval_grid_mode="global"  # or "client"
    # )

    # DEEPSURV CENTRALIZED:
    # user_config = Config(
    #     model="DeepSurv",
    #     centers=[0, 1, 2, 3, 4],
    #     training_mode="centralized",
    #     num_clients=5
    # )

    # # LOCAL TRAINING:
    # user_config = Config(
    #     model="RSF",
    #     centers=[4],
    #     training_mode="local",
    #     num_clients=1,
    #     num_rounds=2,
    #     eval_grid_mode="global"  # or "client"
    # )

    # LOCAL TRAINING DEEPSURV:
    # user_config = Config(
    #     model="DeepSurv",
    #     centers=[4],
    #     training_mode="local",
    #     num_clients=1,
    # )

    main(user_config)
