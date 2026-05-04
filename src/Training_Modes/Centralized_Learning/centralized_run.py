import logging
import os
from datetime import datetime

import numpy as np

from dataset_manager import DatasetManager
from model_manager import ModelManager
from utils import evaluate_model, evaluate_rsf, evaluate_deepsurv


# def _build_global_eval_times_if_needed(config, train_df):
# 	if config.eval_grid_mode != "global" or config.global_eval_times is not None:
# 		return

# 	if "time" not in train_df.columns:
# 		return

# 	train_event_times = train_df.loc[train_df["event"] == 1, "time"].values
# 	if len(train_event_times) > 1:
# 		union_grid = np.unique(train_event_times)
# 		global_grid = np.quantile(union_grid, np.linspace(0.05, 0.95, 100))
# 		config.global_eval_times = global_grid.tolist()


def run_centralized(config):
	logger = logging.getLogger("main")
	logger.info("[Centralized] Starting centralized training pipeline...")

	dm = DatasetManager(config=config, client_idx=0)
	data_bundle = dm.get_centralized_dataloaders()

	global_data = data_bundle["global"]
	per_center = data_bundle.get("per_center", {})

	model_manager = ModelManager(config, client_id=0)
	model_manager.initialize_model()
	model = model_manager.get_model()

	metrics_summary = {"global": None, "per_center": {}}


	if config.model.lower() == "rsf_fedsurf":
		logger.info(f"[Centralized] Training RSF_FedSurF with {config.n_trees_federated} trees")
		model.fit(global_data["X_train"], global_data["y_train"])
		trees = model.estimators_
		n_trees = len(trees)
		logger.info(f"[Centralized] Trained {n_trees} trees")

		metrics_summary["global"] = evaluate_rsf(
			model,
			data={
				"X_test": global_data["X_test"],
				"y_test": global_data["y_test"],
				"y_train": global_data["y_train"],
			},
			client_id=0,
			config=config,
		)

		for center, center_data in per_center.items():
			metrics_summary["per_center"][center] = evaluate_rsf(
				model,
				data={
					"X_test": center_data["X_test"],
					"y_test": center_data["y_test"],
					"y_train": global_data["y_train"],
				},
				client_id=center,
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
				"client_id": "centralized_global",
				"c_index": metrics_summary["global"]["C-index"],
				"auc": metrics_summary["global"]["AUC"],
				"ibs": metrics_summary["global"]["IBS"],
			},
		)

		for center, metrics in metrics_summary["per_center"].items():
			append_metrics_to_csv(
				csv_path,
				{
					"timestamp": datetime.now().isoformat(),
					"run_id": run_id,
					"client_id": f"centralized_{center}",
					"c_index": metrics["C-index"],
					"auc": metrics["AUC"],
					"ibs": metrics["IBS"],
				},
			)

		logger.info("[Centralized] RSF_FedSurF evaluation finished.")
		return metrics_summary

	if config.model.lower() == "deepsurv":
		logger.info(f"[Centralized] Training DeepSurv for {config.num_epochs} epochs with validation")
		
		# Setup client logs directory
		run_id = os.environ.get("RUN_ID", "unknown")
		log_dir = os.path.join(config.experiment_dir, "client_logs")
		os.makedirs(log_dir, exist_ok=True)
		log_file = os.path.join(log_dir, "centralized_training.log")
		
		logger.info(f"[Centralized] Training logs will be saved to: {log_file}")
		
		# Add run separator to log file
		with open(log_file, 'a') as f:
			f.write(f"\n{'='*80}\n")
			f.write(f"RUN {run_id} - CENTRALIZED TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			f.write(f"{'='*80}\n")
		
		# Train with validation data for early stopping
		model.fit(
			global_data["X_train"], 
			global_data["y_train"],
			X_val=global_data["X_val"],
			y_val=global_data["y_val"],
			verbose=True,
			client_id="centralized",
			log_file=log_file
		)
		logger.info(f"[Centralized] DeepSurv training completed")

		metrics_summary["global"] = evaluate_deepsurv(
			model,
			data={
				"X_test": global_data["X_test"],
				"y_test": global_data["y_test"],
				"y_train": global_data["y_train"],
			},
			client_id=0,
			config=config,
		)

		for center, center_data in per_center.items():
			metrics_summary["per_center"][center] = evaluate_deepsurv(
				model,
				data={
					"X_test": center_data["X_test"],
					"y_test": center_data["y_test"],
					"y_train": global_data["y_train"],
				},
				client_id=center,
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
				"client_id": "centralized_global",
				"c_index": metrics_summary["global"]["C-index"],
				"auc": metrics_summary["global"].get("AUC", float('nan')),
				"ibs": metrics_summary["global"].get("IBS", float('nan')),
			},
		)

		for center, metrics in metrics_summary["per_center"].items():
			append_metrics_to_csv(
				csv_path,
				{
					"timestamp": datetime.now().isoformat(),
					"run_id": run_id,
					"client_id": f"centralized_{center}",
					"c_index": metrics["C-index"],
					"auc": metrics.get("AUC", float('nan')),
					"ibs": metrics.get("IBS", float('nan')),
				},
			)

		logger.info("[Centralized] DeepSurv evaluation finished.")
		return metrics_summary

	if config.model.lower() == "coxph":
		logger.info(f"[Centralized] Training CoxPH for {config.num_epochs} epochs with validation")

		run_id = os.environ.get("RUN_ID", "unknown")
		log_dir = os.path.join(config.experiment_dir, "client_logs")
		os.makedirs(log_dir, exist_ok=True)
		log_file = os.path.join(log_dir, "centralized_coxph_training.log")

		logger.info(f"[Centralized] Training logs will be saved to: {log_file}")

		with open(log_file, 'a') as f:
			f.write(f"\n{'='*80}\n")
			f.write(f"RUN {run_id} - CENTRALIZED COXPH TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			f.write(f"{'='*80}\n")

		model.fit(
			global_data["X_train"],
			global_data["y_train"],
			X_val=global_data["X_val"],
			y_val=global_data["y_val"],
			verbose=True,
			client_id="centralized",
			log_file=log_file
		)
		logger.info("[Centralized] CoxPH training completed")

		metrics_summary["global"] = evaluate_deepsurv(
			model,
			data={
				"X_test": global_data["X_test"],
				"y_test": global_data["y_test"],
				"y_train": global_data["y_train"],
			},
			client_id=0,
			config=config,
		)

		for center, center_data in per_center.items():
			metrics_summary["per_center"][center] = evaluate_deepsurv(
				model,
				data={
					"X_test": center_data["X_test"],
					"y_test": center_data["y_test"],
					"y_train": global_data["y_train"],
				},
				client_id=center,
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
				"client_id": "centralized_global",
				"c_index": metrics_summary["global"]["C-index"],
				"auc": metrics_summary["global"].get("AUC", float('nan')),
				"ibs": metrics_summary["global"].get("IBS", float('nan')),
			},
		)

		for center, metrics in metrics_summary["per_center"].items():
			append_metrics_to_csv(
				csv_path,
				{
					"timestamp": datetime.now().isoformat(),
					"run_id": run_id,
					"client_id": f"centralized_{center}",
					"c_index": metrics["C-index"],
					"auc": metrics.get("AUC", float('nan')),
					"ibs": metrics.get("IBS", float('nan')),
				},
			)

		logger.info("[Centralized] CoxPH evaluation finished.")
		return metrics_summary

	raise NotImplementedError(
		f"Centralized training not implemented for model: {config.model}"
	)
