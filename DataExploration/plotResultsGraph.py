# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np



# Data for all strategies at rounds 10, 20, 30
clients = ['C0', 'C1', 'C2', 'C3', 'C4']
strategies = ['FedAvg', 'FedProx', 'FedAdam']

# C-Index means and stds
c_index_means = {
	10: {
		'FedAvg':  [0.818, 0.678, 0.712, 0.520, 0.929],
		'FedProx': [0.822, 0.715, 0.701, 0.501, 0.950],
		'FedAdam':[0.585, 0.443, 0.487, 0.522, 0.475],
	},
	20: {
		'FedAvg':  [0.815, 0.715, 0.671, 0.524, 0.939],
		'FedProx': [0.817, 0.719, 0.670, 0.529, 0.943],
		'FedAdam':[0.603, 0.571, 0.538, 0.458, 0.510],
	},
	30: {
		'FedAvg':  [0.810, 0.647, 0.662, 0.557, 0.943],
		'FedProx': [0.815, 0.693, 0.658, 0.552, 0.946],
		'FedAdam':[0.587, 0.443, 0.490, 0.519, 0.478],
	}
}
c_index_stds = {
	10: {
		'FedAvg':  [0.018, 0.063, 0.030, 0.088, 0.047],
		'FedProx': [0.030, 0.068, 0.050, 0.080, 0.041],
		'FedAdam':[0.124, 0.268, 0.144, 0.142, 0.239],
	},
	20: {
		'FedAvg':  [0.015, 0.051, 0.030, 0.076, 0.045],
		'FedProx': [0.016, 0.078, 0.027, 0.074, 0.045],
		'FedAdam':[0.115, 0.277, 0.103, 0.139, 0.192],
	},
	30: {
		'FedAvg':  [0.022, 0.181, 0.027, 0.078, 0.058],
		'FedProx': [0.018, 0.095, 0.029, 0.097, 0.055],
		'FedAdam':[0.126, 0.268, 0.144, 0.148, 0.237],
	}
}

# AUC means and stds
auc_means = {
	10: {
		'FedAvg':  [0.765, 0.645, 0.679, 0.573, 0.963],
		'FedProx': [0.763, 0.717, 0.681, 0.526, 0.967],
		'FedAdam':[0.562, 0.439, 0.500, 0.520, 0.542],
	},
	20: {
		'FedAvg':  [0.762, 0.701, 0.635, 0.558, 0.944],
		'FedProx': [0.759, 0.701, 0.654, 0.557, 0.949],
		'FedAdam':[0.593, 0.554, 0.541, 0.460, 0.635],
	},
	30: {
		'FedAvg':  [0.766, 0.621, 0.647, 0.589, 0.929],
		'FedProx': [0.768, 0.668, 0.638, 0.594, 0.913],
		'FedAdam':[0.563, 0.439, 0.500, 0.519, 0.542],
	}
}
auc_stds = {
	10: {
		'FedAvg':  [0.031, 0.083, 0.038, 0.087, 0.052],
		'FedProx': [0.036, 0.084, 0.066, 0.082, 0.053],
		'FedAdam':[0.116, 0.269, 0.113, 0.129, 0.251],
	},
	20: {
		'FedAvg':  [0.026, 0.068, 0.061, 0.094, 0.066],
		'FedProx': [0.036, 0.108, 0.048, 0.078, 0.066],
		'FedAdam':[0.109, 0.267, 0.065, 0.123, 0.161],
	},
	30: {
		'FedAvg':  [0.045, 0.222, 0.063, 0.076, 0.107],
		'FedProx': [0.041, 0.144, 0.037, 0.100, 0.102],
		'FedAdam':[0.118, 0.269, 0.113, 0.131, 0.251],
	}
}

# IBS means and stds
ibs_means = {
	10: {
		'FedAvg':  [0.185, 0.260, 0.167, 0.128, 0.046],
		'FedProx': [0.185, 0.240, 0.164, 0.128, 0.046],
		'FedAdam':[0.178, 0.271, 0.194, 0.103, 0.051],
	},
	20: {
		'FedAvg':  [0.181, 0.254, 0.183, 0.125, 0.048],
		'FedProx': [0.179, 0.256, 0.184, 0.123, 0.048],
		'FedAdam':[0.177, 0.269, 0.193, 0.103, 0.051],
	},
	30: {
		'FedAvg':  [0.177, 0.275, 0.182, 0.125, 0.054],
		'FedProx': [0.181, 0.270, 0.184, 0.130, 0.051],
		'FedAdam':[0.178, 0.271, 0.194, 0.103, 0.051],
	}
}
ibs_stds = {
	10: {
		'FedAvg':  [0.026, 0.044, 0.010, 0.022, 0.004],
		'FedProx': [0.019, 0.036, 0.017, 0.013, 0.006],
		'FedAdam':[0.001, 0.004, 0.0006, 0.0004, 0.0001],
	},
	20: {
		'FedAvg':  [0.024, 0.047, 0.026, 0.016, 0.004],
		'FedProx': [0.025, 0.055, 0.021, 0.016, 0.005],
		'FedAdam':[0.001, 0.002, 0.0004, 0.0004, 0.00009],
	},
	30: {
		'FedAvg':  [0.020, 0.082, 0.011, 0.018, 0.010],
		'FedProx': [0.021, 0.066, 0.015, 0.017, 0.003],
		'FedAdam':[0.001, 0.004, 0.0006, 0.0004, 0.00009],
	}
}


# Plotting function for strategy comparison at a given round
def plot_strategy_comparison(clients, means_dict, stds_dict, metric_name, round_num):
	plt.figure(figsize=(8, 5))
	for strategy in strategies:
		plt.errorbar(clients, means_dict[round_num][strategy], yerr=stds_dict[round_num][strategy],
					 label=strategy, marker='o', capsize=4)
	plt.xlabel('Client')
	plt.ylabel(metric_name)
	plt.title(f'{metric_name} Comparison at Round {round_num}')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()

# Plot comparisons for all rounds
for round_num in [10, 20, 30]:
	plot_strategy_comparison(clients, c_index_means, c_index_stds, 'C-Index', round_num)
	plot_strategy_comparison(clients, auc_means, auc_stds, 'AUC', round_num)
	plot_strategy_comparison(clients, ibs_means, ibs_stds, 'IBS', round_num)

# Plotting function
def plot_metric(rounds, means, stds, clients, metric_name):
	plt.figure(figsize=(8, 5))
	for i, client in enumerate(clients):
		plt.errorbar(rounds, [means[j][i] for j in range(len(rounds))],
					 yerr=[stds[j][i] for j in range(len(rounds))],
					 label=client, marker='o', capsize=4)
	plt.xlabel('Rounds')
	plt.ylabel(metric_name)
	plt.title(f'{metric_name} per Client')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()

# Plot C-Index
plot_metric(rounds, c_index_means, c_index_stds, clients, 'C-Index')
# Plot AUC
plot_metric(rounds, auc_means, auc_stds, clients, 'AUC')
# Plot IBS
plot_metric(rounds, ibs_means, ibs_stds, clients, 'IBS')
