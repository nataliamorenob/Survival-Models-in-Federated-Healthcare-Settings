from config import get_default_config
from Training_Modes.Federated_Learning.client import FLClient
from utils import train, test
from Models.CoxPH import fit_cox_model
from config import Config
from flwr.simulation import start_simulation

def main(config: Config):
    

    # Initialize the model
    if config["model"] == "coxph":
        from lifelines import CoxPHFitter
        model = CoxPHFitter()

    # Select training mode
    if config["training_mode"] == "federated":
        # Federated Learning Simulation
        def client_fn(cid):
            return FLClient(model, train_data, test_data, config)

        start_simulation(
            client_fn=client_fn,
            num_clients=10,  # Number of simulated clients
            config={"num_rounds": config["num_rounds"]}
        )

    elif config["training_mode"] == "centralized":
        # Centralized Training
        model = train(model, train_data, config)
        metrics = test(model, test_data, config)
        print("Centralized Training Metrics:", metrics)

    elif config["training_mode"] == "local":
        # Local Training
        model = train(model, train_data, config)
        metrics = test(model, test_data, config)
        print("Local Training Metrics:", metrics)

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

    # Call main function with user configuration
    main(user_config)