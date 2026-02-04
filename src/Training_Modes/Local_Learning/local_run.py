import logging
import os
from datetime import datetime

from dataset_manager import DatasetManager
from model_manager import ModelManager
from utils import evaluate_model, evaluate_rsf
from Exps_runs_randomness.utils_results import append_metrics_to_csv


def run_local(config):
	logger = logging.getLogger("main")
	logger.info("[Local] Starting local training pipeline...")

	dm = DatasetManager(config=config, client_idx=0)
	data = dm.get_local_dataloaders()

	model_manager = ModelManager(config, client_id=0)
	model_manager.initialize_model()
	model = model_manager.get_model()

	# if config.model.lower() == "coxph":
	# 	trained_model = model(
	# 		data["train"],
	# 		config=config,
	# 		client_id=0,
	# 		duration_col="time",
	# 		event_col="event",
	# 		init_params=None,
	# 	)

	# 	metrics = evaluate_model(
	# 		trained_model,
	# 		data["test"],
	# 		config,
	# 		train_data=data["train"],
	# 		client_id=0,
	# 	)
	# 	logger.info("[Local] CoxPH evaluation finished.")
	# 	return metrics

	# if config.model.lower() == "slr":
	# 	X_train = data["train"].drop(columns=["event"])
	# 	y_train = data["train"]["event"]
	# 	model.fit(X_train, y_train)

	# 	metrics = evaluate_model(
	# 		model,
	# 		data["test"],
	# 		config,
	# 		train_data=data["train"],
	# 		client_id=0,
	# 	)
	# 	logger.info("[Local] SLR evaluation finished.")
	# 	return metrics

	if config.model.lower() == "rsf":
		logger.info(f"[Local] Training RSF with {config.n_trees_federated} trees")
		model.fit(data["X_train"], data["y_train"])
		trees = model.estimators_
		logger.info(f"[Local] Trained {len(trees)} trees")

		metrics = evaluate_rsf(
			model,
			data={
				"X_test": data["X_test"],
				"y_test": data["y_test"],
				"y_train": data["y_train"],
			},
			client_id=0,
			config=config,
		)

		run_id = os.environ.get("RUN_ID", "unknown")
		csv_path = os.environ.get(
			"OUTPUT_CSV",
			os.path.join(
				os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
				"results_randomness_exps",
				f"run_{run_id}.csv",
			),
		)

		append_metrics_to_csv(
			csv_path,
			{
				"timestamp": datetime.now().isoformat(),
				"run_id": run_id,
				"client_id": "local_0",
				"c_index": metrics["C-index"],
				"auc": metrics["AUC"],
				"ibs": metrics["IBS"],
			},
		)
		logger.info("[Local] RSF evaluation finished.")
		return metrics

	raise NotImplementedError(
		f"Local training not implemented for model: {config.model}"
	)
