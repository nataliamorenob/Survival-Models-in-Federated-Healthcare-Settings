"""
Visualization script for DeepSurv Federated Learning Results
Generates comprehensive plots to analyze performance across different configurations
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# ============================================================================
# DATA: Results from federated learning experiments
# ============================================================================

# Results structure: {num_clients: {strategy: {rounds: {metric: {client: (mean, std)}}}}}

results_3_clients = {
    'FedAvg': {
        10: {
            'C-Index': {'C0': (0.772, 0.062), 'C1': (0.659, 0.103), 'C2': (0.720, 0.043)},
            'AUC': {'C0': (0.705, 0.074), 'C1': (0.615, 0.149), 'C2': (0.732, 0.061)},
            'IBS': {'C0': (0.214, 0.039), 'C1': (0.301, 0.074), 'C2': (0.162, 0.016)}
        },
        20: {
            'C-Index': {'C0': (0.790, 0.048), 'C1': (0.659, 0.128), 'C2': (0.718, 0.052)},
            'AUC': {'C0': (0.727, 0.058), 'C1': (0.597, 0.174), 'C2': (0.728, 0.070)},
            'IBS': {'C0': (0.223, 0.037), 'C1': (0.344, 0.122), 'C2': (0.172, 0.023)}
        },
        30: {
            'C-Index': {'C0': (0.780, 0.037), 'C1': (0.662, 0.107), 'C2': (0.710, 0.051)},
            'AUC': {'C0': (0.706, 0.044), 'C1': (0.596, 0.149), 'C2': (0.721, 0.054)},
            'IBS': {'C0': (0.229, 0.028), 'C1': (0.343, 0.119), 'C2': (0.180, 0.020)}
        }
    },
    'FedProx': {
        10: {
            'C-Index': {'C0': (0.777, 0.062), 'C1': (0.647, 0.091), 'C2': (0.718, 0.044)},
            'AUC': {'C0': (0.711, 0.065), 'C1': (0.598, 0.126), 'C2': (0.731, 0.063)},
            'IBS': {'C0': (0.212, 0.034), 'C1': (0.304, 0.075), 'C2': (0.162, 0.016)}
        },
        20: {
            'C-Index': {'C0': (0.782, 0.048), 'C1': (0.666, 0.101), 'C2': (0.723, 0.030)},
            'AUC': {'C0': (0.709, 0.061), 'C1': (0.607, 0.155), 'C2': (0.724, 0.052)},
            'IBS': {'C0': (0.222, 0.036), 'C1': (0.333, 0.114), 'C2': (0.174, 0.021)}
        },
        30: {
            'C-Index': {'C0': (0.782, 0.036), 'C1': (0.670, 0.098), 'C2': (0.712, 0.052)},
            'AUC': {'C0': (0.708, 0.042), 'C1': (0.606, 0.157), 'C2': (0.723, 0.064)},
            'IBS': {'C0': (0.229, 0.026), 'C1': (0.350, 0.126), 'C2': (0.180, 0.017)}
        }
    },
    'FedAdam': {
        10: {
            'C-Index': {'C0': (0.584, 0.124), 'C1': (0.443, 0.268), 'C2': (0.489, 0.147)},
            'AUC': {'C0': (0.562, 0.117), 'C1': (0.439, 0.269), 'C2': (0.501, 0.114)},
            'IBS': {'C0': (0.178, 0.001), 'C1': (0.271, 0.004), 'C2': (0.194, 0.0006)}
        },
        20: {
            'C-Index': {'C0': (0.587, 0.126), 'C1': (0.443, 0.268), 'C2': (0.490, 0.145)},
            'AUC': {'C0': (0.563, 0.118), 'C1': (0.439, 0.269), 'C2': (0.501, 0.114)},
            'IBS': {'C0': (0.178, 0.001), 'C1': (0.271, 0.004), 'C2': (0.194, 0.0006)}
        },
        30: {
            'C-Index': {'C0': (0.587, 0.126), 'C1': (0.443, 0.268), 'C2': (0.490, 0.145)},
            'AUC': {'C0': (0.563, 0.119), 'C1': (0.439, 0.269), 'C2': (0.501, 0.114)},
            'IBS': {'C0': (0.178, 0.001), 'C1': (0.271, 0.004), 'C2': (0.194, 0.0006)}
        }
    }
}

results_4_clients = {
    'FedAvg': {
        10: {
            'C-Index': {'C0': (0.815, 0.017), 'C1': (0.715, 0.055), 'C2': (0.725, 0.039), 'C3': (0.506, 0.043)},
            'AUC': {'C0': (0.750, 0.034), 'C1': (0.684, 0.078), 'C2': (0.717, 0.028), 'C3': (0.545, 0.035)},
            'IBS': {'C0': (0.183, 0.022), 'C1': (0.266, 0.056), 'C2': (0.151, 0.009), 'C3': (0.113, 0.014)}
        },
        20: {
            'C-Index': {'C0': (0.822, 0.021), 'C1': (0.712, 0.055), 'C2': (0.722, 0.050), 'C3': (0.512, 0.090)},
            'AUC': {'C0': (0.774, 0.045), 'C1': (0.665, 0.077), 'C2': (0.726, 0.022), 'C3': (0.535, 0.071)},
            'IBS': {'C0': (0.181, 0.029), 'C1': (0.287, 0.069), 'C2': (0.156, 0.014), 'C3': (0.116, 0.015)}
        },
        30: {
            'C-Index': {'C0': (0.827, 0.016), 'C1': (0.696, 0.068), 'C2': (0.732, 0.060), 'C3': (0.524, 0.082)},
            'AUC': {'C0': (0.774, 0.046), 'C1': (0.629, 0.098), 'C2': (0.730, 0.044), 'C3': (0.570, 0.083)},
            'IBS': {'C0': (0.185, 0.032), 'C1': (0.296, 0.074), 'C2': (0.161, 0.018), 'C3': (0.112, 0.013)}
        }
    },
    'FedAdam': {
        10: {
            'C-Index': {'C0': (0.584, 0.123), 'C1': (0.443, 0.268), 'C2': (0.487, 0.144), 'C3': (0.522, 0.142)},
            'AUC': {'C0': (0.562, 0.116), 'C1': (0.439, 0.269), 'C2': (0.500, 0.113), 'C3': (0.520, 0.129)},
            'IBS': {'C0': (0.178, 0.001), 'C1': (0.271, 0.004), 'C2': (0.194, 0.0006), 'C3': (0.103, 0.0004)}
        },
        20: {
            'C-Index': {'C0': (0.586, 0.125), 'C1': (0.443, 0.268), 'C2': (0.489, 0.145), 'C3': (0.522, 0.142)},
            'AUC': {'C0': (0.563, 0.118), 'C1': (0.439, 0.269), 'C2': (0.500, 0.113), 'C3': (0.520, 0.129)},
            'IBS': {'C0': (0.178, 0.001), 'C1': (0.271, 0.004), 'C2': (0.194, 0.0006), 'C3': (0.103, 0.0004)}
        },
        30: {
            'C-Index': {'C0': (0.587, 0.125), 'C1': (0.443, 0.268), 'C2': (0.492, 0.144), 'C3': (0.522, 0.142)},
            'AUC': {'C0': (0.563, 0.118), 'C1': (0.439, 0.269), 'C2': (0.501, 0.114), 'C3': (0.520, 0.129)},
            'IBS': {'C0': (0.178, 0.001), 'C1': (0.271, 0.004), 'C2': (0.194, 0.0006), 'C3': (0.103, 0.0004)}
        }
    },
    'FedProx': {
        10: {
            'C-Index': {'C0': (0.816, 0.017), 'C1': (0.715, 0.058), 'C2': (0.729, 0.038), 'C3': (0.491, 0.057)},
            'AUC': {'C0': (0.755, 0.028), 'C1': (0.685, 0.079), 'C2': (0.728, 0.030), 'C3': (0.532, 0.046)},
            'IBS': {'C0': (0.181, 0.021), 'C1': (0.264, 0.055), 'C2': (0.150, 0.008), 'C3': (0.115, 0.014)}
        },
        20: {
            'C-Index': {'C0': (0.818, 0.020), 'C1': (0.700, 0.087), 'C2': (0.722, 0.054), 'C3': (0.564, 0.062)},
            'AUC': {'C0': (0.757, 0.045), 'C1': (0.650, 0.132), 'C2': (0.729, 0.026), 'C3': (0.602, 0.034)},
            'IBS': {'C0': (0.186, 0.027), 'C1': (0.288, 0.087), 'C2': (0.157, 0.014), 'C3': (0.115, 0.013)}
        },
        30: {
            'C-Index': {'C0': (0.824, 0.020), 'C1': (0.693, 0.077), 'C2': (0.728, 0.050), 'C3': (0.519, 0.091)},
            'AUC': {'C0': (0.773, 0.046), 'C1': (0.637, 0.116), 'C2': (0.729, 0.041), 'C3': (0.574, 0.069)},
            'IBS': {'C0': (0.182, 0.029), 'C1': (0.299, 0.082), 'C2': (0.161, 0.015), 'C3': (0.111, 0.013)}
        }
    }
}

results_5_clients = {
    'FedAvg': {
        10: {
            'C-Index': {'C0': (0.818, 0.018), 'C1': (0.678, 0.063), 'C2': (0.712, 0.030), 'C3': (0.520, 0.088), 'C4': (0.929, 0.047)},
            'AUC': {'C0': (0.765, 0.031), 'C1': (0.645, 0.083), 'C2': (0.679, 0.038), 'C3': (0.573, 0.087), 'C4': (0.963, 0.052)},
            'IBS': {'C0': (0.185, 0.026), 'C1': (0.260, 0.044), 'C2': (0.167, 0.010), 'C3': (0.128, 0.022), 'C4': (0.046, 0.004)}
        },
        20: {
            'C-Index': {'C0': (0.815, 0.015), 'C1': (0.715, 0.051), 'C2': (0.671, 0.030), 'C3': (0.524, 0.076), 'C4': (0.939, 0.045)},
            'AUC': {'C0': (0.762, 0.026), 'C1': (0.701, 0.068), 'C2': (0.635, 0.061), 'C3': (0.558, 0.094), 'C4': (0.944, 0.064)},
            'IBS': {'C0': (0.181, 0.024), 'C1': (0.254, 0.047), 'C2': (0.183, 0.026), 'C3': (0.125, 0.016), 'C4': (0.048, 0.004)}
        },
        30: {
            'C-Index': {'C0': (0.810, 0.022), 'C1': (0.647, 0.181), 'C2': (0.662, 0.027), 'C3': (0.557, 0.078), 'C4': (0.943, 0.058)},
            'AUC': {'C0': (0.766, 0.045), 'C1': (0.621, 0.222), 'C2': (0.647, 0.063), 'C3': (0.589, 0.076), 'C4': (0.929, 0.107)},
            'IBS': {'C0': (0.177, 0.020), 'C1': (0.275, 0.082), 'C2': (0.182, 0.011), 'C3': (0.125, 0.018), 'C4': (0.054, 0.010)}
        }
    },
    'FedAdam': {
        10: {
            'C-Index': {'C0': (0.585, 0.124), 'C1': (0.443, 0.268), 'C2': (0.487, 0.144), 'C3': (0.522, 0.142), 'C4': (0.475, 0.239)},
            'AUC': {'C0': (0.562, 0.116), 'C1': (0.439, 0.269), 'C2': (0.500, 0.113), 'C3': (0.520, 0.129), 'C4': (0.542, 0.251)},
            'IBS': {'C0': (0.178, 0.001), 'C1': (0.271, 0.004), 'C2': (0.194, 0.0006), 'C3': (0.103, 0.0004), 'C4': (0.051, 0.0001)}
        },
        20: {
            'C-Index': {'C0': (0.603, 0.115), 'C1': (0.571, 0.277), 'C2': (0.538, 0.103), 'C3': (0.458, 0.139), 'C4': (0.510, 0.192)},
            'AUC': {'C0': (0.593, 0.109), 'C1': (0.554, 0.267), 'C2': (0.541, 0.065), 'C3': (0.460, 0.123), 'C4': (0.635, 0.161)},
            'IBS': {'C0': (0.177, 0.001), 'C1': (0.269, 0.002), 'C2': (0.193, 0.0004), 'C3': (0.103, 0.0002), 'C4': (0.051, 0.00009)}
        },
        30: {
            'C-Index': {'C0': (0.587, 0.126), 'C1': (0.443, 0.268), 'C2': (0.490, 0.144), 'C3': (0.519, 0.148), 'C4': (0.478, 0.237)},
            'AUC': {'C0': (0.563, 0.118), 'C1': (0.439, 0.269), 'C2': (0.501, 0.113), 'C3': (0.519, 0.131), 'C4': (0.542, 0.251)},
            'IBS': {'C0': (0.178, 0.001), 'C1': (0.271, 0.004), 'C2': (0.194, 0.0006), 'C3': (0.103, 0.0004), 'C4': (0.051, 0.0001)}
        }
    },
    'FedProx': {
        10: {
            'C-Index': {'C0': (0.822, 0.030), 'C1': (0.715, 0.068), 'C2': (0.701, 0.050), 'C3': (0.501, 0.080), 'C4': (0.950, 0.041)},
            'AUC': {'C0': (0.763, 0.036), 'C1': (0.717, 0.084), 'C2': (0.681, 0.066), 'C3': (0.526, 0.082), 'C4': (0.967, 0.053)},
            'IBS': {'C0': (0.185, 0.019), 'C1': (0.240, 0.036), 'C2': (0.164, 0.017), 'C3': (0.128, 0.013), 'C4': (0.046, 0.006)}
        },
        20: {
            'C-Index': {'C0': (0.817, 0.016), 'C1': (0.719, 0.078), 'C2': (0.670, 0.027), 'C3': (0.529, 0.074), 'C4': (0.943, 0.045)},
            'AUC': {'C0': (0.759, 0.036), 'C1': (0.701, 0.108), 'C2': (0.654, 0.048), 'C3': (0.557, 0.078), 'C4': (0.949, 0.066)},
            'IBS': {'C0': (0.179, 0.025), 'C1': (0.256, 0.055), 'C2': (0.184, 0.021), 'C3': (0.123, 0.016), 'C4': (0.048, 0.005)}
        },
        30: {
            'C-Index': {'C0': (0.815, 0.018), 'C1': (0.693, 0.095), 'C2': (0.658, 0.029), 'C3': (0.552, 0.097), 'C4': (0.946, 0.055)},
            'AUC': {'C0': (0.768, 0.041), 'C1': (0.668, 0.144), 'C2': (0.638, 0.037), 'C3': (0.594, 0.100), 'C4': (0.913, 0.102)},
            'IBS': {'C0': (0.181, 0.021), 'C1': (0.270, 0.066), 'C2': (0.184, 0.015), 'C3': (0.130, 0.017), 'C4': (0.051, 0.003)}
        }
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_average_performance(results_dict, strategy, rounds, metric):
    """Calculate average performance across all clients"""
    client_means = [results_dict[strategy][rounds][metric][client][0] 
                    for client in results_dict[strategy][rounds][metric]]
    return np.mean(client_means)

def get_average_std(results_dict, strategy, rounds, metric):
    """Calculate average standard deviation across all clients"""
    client_stds = [results_dict[strategy][rounds][metric][client][1] 
                   for client in results_dict[strategy][rounds][metric]]
    return np.mean(client_stds)

# ============================================================================
# PLOT 1: Individual client trajectories across rounds
# ============================================================================

def plot_individual_client_trajectories(save_path='client_trajectories.png'):
    """
    Line plots showing how EACH CLIENT evolves across rounds for different strategies
    This reveals client heterogeneity and identifies problematic clients like C3
    """
    metrics = ['C-Index', 'AUC', 'IBS']
    strategies = ['FedAvg', 'FedProx', 'FedAdam']
    rounds_list = [10, 20, 30]
    
    # Use 5 clients for most comprehensive view
    results_dict = results_5_clients
    clients = ['C0', 'C1', 'C2', 'C3', 'C4']
    
    # Color palette for clients
    client_colors = {'C0': '#E63946', 'C1': '#F1A208', 'C2': '#2A9D8F', 
                     'C3': '#264653', 'C4': '#9D4EDD'}
    markers = ['o', 's', '^', 'D', 'v']
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    
    for row_idx, strategy in enumerate(strategies):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            for client_idx, client in enumerate(clients):
                values = []
                for rounds in rounds_list:
                    value = results_dict[strategy][rounds][metric][client][0]
                    values.append(value)
                
                ax.plot(rounds_list, values, marker=markers[client_idx], 
                       color=client_colors[client], linewidth=2.5, 
                       markersize=9, label=client, alpha=0.85)
            
            ax.set_xlabel('Rounds', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{strategy} - {metric}', fontsize=12, fontweight='bold')
            ax.set_xticks(rounds_list)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add reference lines
            if metric == 'C-Index' or metric == 'AUC':
                ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, 
                          alpha=0.4, label='Random' if col_idx == 0 else '')
            
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc='best', framealpha=0.95, fontsize=10)
    
    plt.suptitle('Individual Client Performance Trajectories (5 Clients)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

# ============================================================================
# PLOT 2: Client-by-client strategy comparison
# ============================================================================

def plot_client_strategy_comparison(save_path='client_strategy_comparison.png'):
    """
    Compare strategies for EACH CLIENT separately across all metrics
    Shows which strategy works best for each individual client
    """
    metrics = ['C-Index', 'AUC', 'IBS']
    strategies = ['FedAvg', 'FedProx', 'FedAdam']
    
    # Use 5 clients at 30 rounds
    results_dict = results_5_clients
    rounds = 30
    clients = ['C0', 'C1', 'C2', 'C3', 'C4']
    
    colors = {'FedAvg': '#2E86AB', 'FedProx': '#A23B72', 'FedAdam': '#F18F01'}
    
    fig, axes = plt.subplots(len(clients), len(metrics), figsize=(18, 16))
    
    for row_idx, client in enumerate(clients):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            values = []
            for strategy in strategies:
                value = results_dict[strategy][rounds][metric][client][0]
                values.append(value)
            
            x_pos = np.arange(len(strategies))
            bars = ax.bar(x_pos, values, color=[colors[s] for s in strategies],
                         alpha=0.75, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Add reference line for random guessing
            if metric in ['C-Index', 'AUC']:
                ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, 
                          alpha=0.5, label='Random' if row_idx == 0 else '')
                if row_idx == 0:
                    ax.legend(fontsize=8)
            
            ax.set_ylabel(metric if col_idx == 0 else '', fontsize=10, fontweight='bold')
            ax.set_title(f'{client} - {metric}' if row_idx == 0 else '', 
                        fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(strategies, rotation=0, fontsize=9)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add client label on left
            if col_idx == 0:
                ax.text(-0.4, 0.5, client, transform=ax.transAxes,
                       fontsize=13, fontweight='bold', va='center', 
                       rotation=90, color='#264653')
    
    plt.suptitle('Strategy Comparison per Client (30 Rounds, 5 Clients)', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

# ============================================================================
# PLOT 3: Heatmap - Client performance patterns
# ============================================================================

def plot_client_performance_heatmap(save_path='client_performance_heatmap.png'):
    """
    Heatmaps showing performance of EACH CLIENT across strategies and rounds
    Separate heatmap for each metric - makes it easy to spot problematic clients
    """
    metrics = ['C-Index', 'AUC', 'IBS']
    strategies = ['FedAvg', 'FedProx', 'FedAdam']
    rounds_list = [10, 20, 30]
    
    # Use 5 clients
    results_dict = results_5_clients
    clients = ['C0', 'C1', 'C2', 'C3', 'C4']
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    for idx, metric in enumerate(metrics):
        # Prepare data matrix: rows=clients, columns=strategy-round combinations
        config_labels = []
        data_matrix = []
        
        for client in clients:
            row_data = []
            for strategy in strategies:
                for rounds in rounds_list:
                    value = results_dict[strategy][rounds][metric][client][0]
                    row_data.append(value)
                    if client == clients[0]:  # Only add labels once
                        config_labels.append(f'{strategy}\n{rounds}R')
            data_matrix.append(row_data)
        
        # Create heatmap
        ax = axes[idx]
        
        # Choose colormap based on metric
        if metric == 'IBS':
            cmap = 'RdYlGn_r'  # Red for high (bad), green for low (good)
            vmin, vmax = 0, 0.35
        else:
            cmap = 'RdYlGn'  # Red for low (bad), green for high (good)
            vmin, vmax = 0.4, 1.0
        
        im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set ticks
        ax.set_yticks(np.arange(len(clients)))
        ax.set_yticklabels(clients, fontsize=11, fontweight='bold')
        ax.set_xticks(np.arange(len(config_labels)))
        ax.set_xticklabels(config_labels, rotation=0, ha='center', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric, rotation=270, labelpad=20, fontweight='bold', fontsize=11)
        
        # Annotate cells with values
        for i in range(len(clients)):
            for j in range(len(config_labels)):
                value = data_matrix[i][j]
                # Make text white for dark cells, black for light cells
                text_color = "white" if (metric == 'IBS' and value > 0.2) or \
                                       (metric != 'IBS' and value < 0.6) else "black"
                text = ax.text(j, i, f'{value:.2f}',
                             ha="center", va="center", color=text_color, 
                             fontsize=8, fontweight='bold')
        
        ax.set_title(f'{metric} - Per Client Performance', fontsize=13, fontweight='bold')
        ax.set_xlabel('Strategy & Rounds', fontsize=11, fontweight='bold')
        ax.set_ylabel('Client', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

# ============================================================================
# PLOT 4: Client ranking across configurations
# ============================================================================

def plot_client_ranking(save_path='client_ranking.png'):
    """
    Shows the ranking of clients by performance across different metrics
    Helps identify consistently strong/weak clients
    """
    metrics = ['C-Index', 'AUC', 'IBS']
    strategies = ['FedAvg', 'FedProx', 'FedAdam']
    
    # Use 5 clients at 30 rounds
    results_dict = results_5_clients
    rounds = 30
    clients = ['C0', 'C1', 'C2', 'C3', 'C4']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for col_idx, metric in enumerate(metrics):
        ax = axes[col_idx]
        
        # Collect data for all strategies
        all_data = {client: [] for client in clients}
        strategy_labels = []
        
        for strategy in strategies:
            for client in clients:
                value = results_dict[strategy][rounds][metric][client][0]
                all_data[client].append(value)
            strategy_labels.append(strategy)
        
        # Sort clients by average performance
        if metric == 'IBS':
            # For IBS, lower is better
            client_order = sorted(clients, key=lambda c: np.mean(all_data[c]))
        else:
            # For C-Index and AUC, higher is better
            client_order = sorted(clients, key=lambda c: np.mean(all_data[c]), reverse=True)
        
        # Plot
        x = np.arange(len(strategies))
        width = 0.16
        
        colors = {'C0': '#E63946', 'C1': '#F1A208', 'C2': '#2A9D8F', 
                 'C3': '#264653', 'C4': '#9D4EDD'}
        
        for i, client in enumerate(client_order):
            offset = width * (i - 2)
            bars = ax.bar(x + offset, all_data[client], width, label=client,
                         color=colors[client], alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} - Client Ranking (30 Rounds)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.legend(title='Best→Worst', loc='best', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add reference line
        if metric in ['C-Index', 'AUC']:
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

# ============================================================================
# PLOT 5: Per-client radar plots
# ============================================================================

def plot_per_client_radar(save_path='per_client_radar.png'):
    """
    Radar plots showing multi-metric performance for EACH CLIENT
    Reveals which clients have balanced performance vs which are problematic
    """
    from math import pi
    
    strategies = ['FedAvg', 'FedProx', 'FedAdam']
    colors = {'FedAvg': '#2E86AB', 'FedProx': '#A23B72', 'FedAdam': '#F18F01'}
    
    # Use 5 clients at 30 rounds
    results_dict = results_5_clients
    rounds = 30
    clients = ['C0', 'C1', 'C2', 'C3', 'C4']
    
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), subplot_kw=dict(projection='polar'))
    
    for idx, client in enumerate(clients):
        ax = axes[idx]
        
        # Metrics for radar chart
        categories = ['C-Index', 'AUC', '1-IBS']
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        for strategy in strategies:
            values = []
            
            # C-Index
            c_index = results_dict[strategy][rounds]['C-Index'][client][0]
            values.append(c_index)
            
            # AUC
            auc = results_dict[strategy][rounds]['AUC'][client][0]
            values.append(auc)
            
            # 1-IBS (invert so higher is better)
            ibs = results_dict[strategy][rounds]['IBS'][client][0]
            values.append(1 - ibs)
            
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2.5, 
                   label=strategy, color=colors[strategy])
            ax.fill(angles, values, alpha=0.15, color=colors[strategy])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title(client, fontsize=13, fontweight='bold', pad=20)
        ax.grid(True)
        
        # Add reference circle at 0.5 (random guessing)
        ax.plot(angles, [0.5] * len(angles), 'r--', linewidth=1.5, alpha=0.4)
        
        if idx == 4:
            ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=10)
    
    plt.suptitle('Per-Client Multi-Metric Performance (30 Rounds)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

# ============================================================================
# PLOT 6: Client consistency analysis
# ============================================================================

def plot_client_consistency(save_path='client_consistency.png'):
    """
    Shows variance in performance across strategies and rounds for each client
    Identifies which clients have stable vs unstable performance
    """
    metrics = ['C-Index', 'AUC', 'IBS']
    strategies = ['FedAvg', 'FedProx', 'FedAdam']
    rounds_list = [10, 20, 30]
    
    # Use 5 clients
    results_dict = results_5_clients
    clients = ['C0', 'C1', 'C2', 'C3', 'C4']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors_clients = {'C0': '#E63946', 'C1': '#F1A208', 'C2': '#2A9D8F', 
                     'C3': '#264653', 'C4': '#9D4EDD'}
    
    for col_idx, metric in enumerate(metrics):
        ax = axes[col_idx]
        
        # For each client, collect all values across strategies and rounds
        client_means = []
        client_stds = []
        client_labels = []
        
        for client in clients:
            all_values = []
            for strategy in strategies:
                for rounds in rounds_list:
                    value = results_dict[strategy][rounds][metric][client][0]
                    all_values.append(value)
            
            client_means.append(np.mean(all_values))
            client_stds.append(np.std(all_values))
            client_labels.append(client)
        
        # Create scatter plot: mean vs std (consistency)
        for i, client in enumerate(clients):
            ax.scatter(client_stds[i], client_means[i], 
                      s=300, alpha=0.7, color=colors_clients[client],
                      edgecolor='black', linewidth=2, label=client)
            ax.text(client_stds[i], client_means[i], client,
                   ha='center', va='center', fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Std Dev (Inconsistency)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric} Mean', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} - Consistency Analysis', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add quadrant labels
        mid_x = np.median(client_stds)
        mid_y = np.median(client_means)
        
        if metric != 'IBS':
            ax.text(0.95, 0.95, 'High & Stable', transform=ax.transAxes,
                   ha='right', va='top', fontsize=10, style='italic',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
            ax.text(0.05, 0.05, 'Low & Stable', transform=ax.transAxes,
                   ha='left', va='bottom', fontsize=10, style='italic',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        else:
            ax.text(0.95, 0.05, 'Low & Stable', transform=ax.transAxes,
                   ha='right', va='bottom', fontsize=10, style='italic',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_all_plots(output_dir='plots_output'):
    """
    Generate all visualization plots focused on INDIVIDUAL CLIENT analysis
    """
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}/")
    
    print("\n" + "="*70)
    print("GENERATING PER-CLIENT DEEPSURV FEDERATED LEARNING VISUALIZATIONS")
    print("="*70 + "\n")
    
    print("Generating plots...")
    print("-" * 70)
    
    plot_individual_client_trajectories(f'{output_dir}/1_client_trajectories.png')
    plot_client_strategy_comparison(f'{output_dir}/2_client_strategy_comparison.png')
    plot_client_performance_heatmap(f'{output_dir}/3_client_performance_heatmap.png')
    plot_client_ranking(f'{output_dir}/4_client_ranking.png')
    plot_per_client_radar(f'{output_dir}/5_per_client_radar.png')
    plot_client_consistency(f'{output_dir}/6_client_consistency.png')
    
    print("-" * 70)
    print(f"\n✅ All plots generated successfully in '{output_dir}/' directory!")
    print("\n" + "="*70)
    print("KEY INSIGHTS TO LOOK FOR (PER-CLIENT FOCUS):")
    print("="*70)
    print("1. Client Trajectories: How does EACH client evolve across rounds?")
    print("2. Client-Strategy Comparison: Which strategy works best for EACH client?")
    print("3. Performance Heatmap: Quickly identify problematic clients (like C3)")
    print("4. Client Ranking: Who are the best/worst performers consistently?")
    print("5. Per-Client Radar: Which clients have balanced vs unbalanced metrics?")
    print("6. Client Consistency: Which clients are stable vs unpredictable?")
    print("="*70)
    print("\n🔍 SPECIAL ATTENTION: Look for Client 3's poor C-Index/AUC but good IBS!")
    print("="*70 + "\n")

if __name__ == "__main__":
    generate_all_plots()
