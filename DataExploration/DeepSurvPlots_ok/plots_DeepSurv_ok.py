"""
Visualization script for DeepSurv Federated Learning Results
Compares different strategies (FedAvg, FedProx, FedAdam) across multiple client configurations (3, 4, 5)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = Path("plots_output_DeepSurv")
OUTPUT_DIR.mkdir(exist_ok=True)

# ========================================
# DATA PREPARATION
# ========================================

# Results data structure
results = {
    # 5 clients configuration
    5: {
        'FedAvg': {
            'rounds': 11,
            'clients': ['C0', 'C1', 'C2', 'C3', 'C4'],
            'c_index': [0.838, 0.776, 0.775, 0.765, 0.875],
            'c_index_std': [0.031, 0.084, 0.065, 0.047, 0.041],
            'auc': [0.823, 0.785, 0.619, 0.846, 0.917],
            'auc_std': [0.021, 0.112, 0.077, 0.055, 0.069],
            'ibs': [0.140, 0.234, 0.199, 0.086, 0.039],
            'ibs_std': [0.010, 0.042, 0.031, 0.006, 0.003]
        },
        'FedProx': {
            'rounds': 11,
            'clients': ['C0', 'C1', 'C2', 'C3', 'C4'],
            'c_index': [0.844, 0.780, 0.770, 0.744, 0.893],
            'c_index_std': [0.032, 0.090, 0.062, 0.050, 0.030],
            'auc': [0.830, 0.795, 0.598, 0.827, 0.947],
            'auc_std': [0.031, 0.106, 0.065, 0.049, 0.027],
            'ibs': [0.137, 0.235, 0.204, 0.088, 0.038],
            'ibs_std': [0.014, 0.042, 0.029, 0.007, 0.003]
        },
        'FedAdam': {
            'rounds': 10,
            'clients': ['C0', 'C1', 'C2', 'C3', 'C4'],
            'c_index': [0.861, 0.776, 0.782, 0.718, 0.882],
            'c_index_std': [0.026, 0.091, 0.024, 0.069, 0.051],
            'auc': [0.842, 0.785, 0.623, 0.758, 0.931],
            'auc_std': [0.033, 0.097, 0.030, 0.092, 0.048],
            'ibs': [0.142, 0.253, 0.226, 0.088, 0.037],
            'ibs_std': [0.025, 0.054, 0.043, 0.007, 0.007]
        }
    },
    # 4 clients configuration
    4: {
        'FedAvg': {
            'rounds': 9,
            'clients': ['C0', 'C1', 'C2', 'C3'],
            'c_index': [0.809, 0.765, 0.794, 0.782],
            'c_index_std': [0.021, 0.103, 0.063, 0.044],
            'auc': [0.785, 0.763, 0.658, 0.877],
            'auc_std': [0.026, 0.127, 0.081, 0.032],
            'ibs': [0.145, 0.229, 0.189, 0.081],
            'ibs_std': [0.014, 0.036, 0.019, 0.005]
        },
        'FedProx': {
            'rounds': 9,
            'clients': ['C0', 'C1', 'C2', 'C3'],
            'c_index': [0.812, 0.761, 0.793, 0.777],
            'c_index_std': [0.016, 0.101, 0.060, 0.040],
            'auc': [0.786, 0.763, 0.654, 0.877],
            'auc_std': [0.022, 0.128, 0.089, 0.030],
            'ibs': [0.145, 0.230, 0.187, 0.082],
            'ibs_std': [0.014, 0.033, 0.018, 0.005]
        },
        'FedAdam': {
            'rounds': 11,
            'clients': ['C0', 'C1', 'C2', 'C3'],
            'c_index': [0.838, 0.761, 0.831, 0.828],
            'c_index_std': [0.028, 0.132, 0.063, 0.062],
            'auc': [0.826, 0.774, 0.733, 0.891],
            'auc_std': [0.037, 0.142, 0.112, 0.040],
            'ibs': [0.137, 0.230, 0.176, 0.076],
            'ibs_std': [0.025, 0.069, 0.036, 0.004]
        }
    },
    # 3 clients configuration
    3: {
        'FedAvg': {
            'rounds': 15,
            'clients': ['C0', 'C1', 'C2'],
            'c_index': [0.810, 0.719, 0.806],
            'c_index_std': [0.021, 0.096, 0.053],
            'auc': [0.788, 0.731, 0.699],
            'auc_std': [0.033, 0.114, 0.076],
            'ibs': [0.154, 0.249, 0.185],
            'ibs_std': [0.018, 0.039, 0.025]
        },
        'FedProx': {
            'rounds': 14,
            'clients': ['C0', 'C1', 'C2'],
            'c_index': [0.803, 0.715, 0.796],
            'c_index_std': [0.019, 0.104, 0.053],
            'auc': [0.779, 0.735, 0.671],
            'auc_std': [0.028, 0.123, 0.091],
            'ibs': [0.155, 0.246, 0.186],
            'ibs_std': [0.015, 0.043, 0.023]
        },
        'FedAdam': {
            'rounds': 7,
            'clients': ['C0', 'C1', 'C2'],
            'c_index': [0.813, 0.715, 0.786],
            'c_index_std': [0.020, 0.114, 0.058],
            'auc': [0.778, 0.727, 0.649],
            'auc_std': [0.020, 0.106, 0.090],
            'ibs': [0.163, 0.247, 0.210],
            'ibs_std': [0.013, 0.033, 0.033]
        }
    }
}

# ========================================
# PLOT 1: C-Index Comparison Across All Configurations
# ========================================

def plot_cindex_comparison():
    """Bar plot comparing C-Index across strategies and client counts"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('C-Index Performance Comparison Across Strategies and Client Configurations', 
                 fontsize=12, fontweight='bold', y=1.02)
    
    strategies = ['FedAvg', 'FedProx', 'FedAdam']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (n_clients, ax) in enumerate(zip([5, 4, 3], axes)):
        positions = np.arange(len(results[n_clients]['FedAvg']['clients']))
        width = 0.25
        
        for i, strategy in enumerate(strategies):
            data = results[n_clients][strategy]
            offset = (i - 1) * width
            bars = ax.bar(positions + offset, data['c_index'], width, 
                         label=f"{strategy} (R={data['rounds']})",
                         color=colors[i], alpha=0.8, 
                         yerr=data['c_index_std'], capsize=5)
            
        ax.set_xlabel('Client', fontsize=12, fontweight='bold')
        ax.set_ylabel('C-Index', fontsize=12, fontweight='bold')
        ax.set_title(f'{n_clients} Clients Configuration', fontsize=13, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(results[n_clients]['FedAvg']['clients'])
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0.6, 1.0])
        
        # Add horizontal line for paper benchmark if needed
        ax.axhline(y=0.789, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Paper (78.9%)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cindex_comparison_all.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'cindex_comparison_all.png'}")
    plt.show()

# ========================================
# PLOT 2: Strategy Performance by Metric (Heatmaps)
# ========================================

def plot_strategy_heatmaps():
    """Heatmaps showing performance across strategies, clients, and metrics"""
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle('Performance Heatmaps: Strategy × Client Configuration', 
                 fontsize=13, fontweight='bold', y=0.995)
    
    metrics = ['c_index', 'auc', 'ibs']
    metric_names = ['C-Index', 'AUC', 'IBS']
    client_configs = [5, 4, 3]
    
    for col_idx, n_clients in enumerate(client_configs):
        for row_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[row_idx, col_idx]
            
            # Prepare data matrix: strategies × clients
            strategies = ['FedAvg', 'FedProx', 'FedAdam']
            matrix_data = []
            for strategy in strategies:
                matrix_data.append(results[n_clients][strategy][metric])
            
            matrix_data = np.array(matrix_data)
            
            # Create heatmap
            if metric == 'ibs':
                cmap = 'RdYlGn_r'  # Lower is better for IBS
                vmin, vmax = 0, 0.3
            else:
                cmap = 'RdYlGn'  # Higher is better
                vmin, vmax = 0.5, 1.0  # Same scale for both c_index and auc
            
            im = ax.imshow(matrix_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(results[n_clients]['FedAvg']['clients'])))
            ax.set_yticks(np.arange(len(strategies)))
            ax.set_xticklabels(results[n_clients]['FedAvg']['clients'])
            ax.set_yticklabels([f"{s} (R={results[n_clients][s]['rounds']})" for s in strategies])
            
            # Add values in cells
            for i in range(len(strategies)):
                for j in range(len(results[n_clients]['FedAvg']['clients'])):
                    text = ax.text(j, i, f'{matrix_data[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=9)
            
            # Labels
            if col_idx == 0:
                ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
            if row_idx == 0:
                ax.set_title(f'{n_clients} Clients', fontsize=13, fontweight='bold')
            if row_idx == 2:
                ax.set_xlabel('Client ID', fontsize=11)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'strategy_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'strategy_heatmaps.png'}")
    plt.show()

# ========================================
# PLOT 3: Average Performance Across All Clients
# ========================================

def plot_average_performance():
    """Compare average performance across all clients for each configuration"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Average Performance Across All Clients', 
                 fontsize=12, fontweight='bold', y=1.02)
    
    metrics = ['c_index', 'auc', 'ibs']
    metric_names = ['C-Index', 'AUC', 'IBS (lower is better)']
    colors = {'FedAvg': '#3498db', 'FedProx': '#e74c3c', 'FedAdam': '#2ecc71'}
    
    for ax, metric, metric_name in zip(axes, metrics, metric_names):
        client_configs = [3, 4, 5]
        x_pos = np.arange(len(client_configs))
        width = 0.25
        
        for i, strategy in enumerate(['FedAvg', 'FedProx', 'FedAdam']):
            means = []
            stds = []
            for n_clients in client_configs:
                data = results[n_clients][strategy][metric]
                means.append(np.mean(data))
                # Propagate uncertainty
                stds.append(np.sqrt(np.sum(np.array(results[n_clients][strategy][f'{metric}_std'])**2)) / len(data))
            
            offset = (i - 1) * width
            bars = ax.bar(x_pos + offset, means, width, label=strategy,
                         color=colors[strategy], alpha=0.8, yerr=stds, capsize=5)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Number of Clients', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(client_configs)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        if metric != 'ibs':
            ax.set_ylim([0.7, 0.9])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'average_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'average_performance.png'}")
    plt.show()

# ========================================
# PLOT 4: Convergence Rounds Analysis
# ========================================

def plot_convergence_rounds():
    """Visualize optimal convergence rounds for each strategy and configuration"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    strategies = ['FedAvg', 'FedProx', 'FedAdam']
    client_configs = [3, 4, 5]
    
    # Prepare data
    data_for_plot = []
    for strategy in strategies:
        for n_clients in client_configs:
            rounds = results[n_clients][strategy]['rounds']
            avg_cindex = np.mean(results[n_clients][strategy]['c_index'])
            data_for_plot.append({
                'Strategy': strategy,
                'Clients': n_clients,
                'Optimal Rounds': rounds,
                'Avg C-Index': avg_cindex
            })
    
    df = pd.DataFrame(data_for_plot)
    
    # Create grouped bar plot
    x = np.arange(len(client_configs))
    width = 0.25
    colors = {'FedAvg': '#3498db', 'FedProx': '#e74c3c', 'FedAdam': '#2ecc71'}
    
    for i, strategy in enumerate(strategies):
        strategy_data = df[df['Strategy'] == strategy]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, strategy_data['Optimal Rounds'], width,
                     label=strategy, color=colors[strategy], alpha=0.8)
        
        # Add value labels
        for bar, (_, row) in zip(bars, strategy_data.iterrows()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({row["Avg C-Index"]:.3f})',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Number of Clients', fontsize=13, fontweight='bold')
    ax.set_ylabel('Optimal Convergence Rounds', fontsize=13, fontweight='bold')
    ax.set_title('Convergence Analysis: Optimal Rounds and Average C-Index', 
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(client_configs)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'convergence_rounds.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'convergence_rounds.png'}")
    plt.show()

# ========================================
# PLOT 5: Variance Analysis
# ========================================

def plot_variance_analysis():
    """Analyze and visualize variance across clients (stability metric)"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Variance Analysis: Performance Stability Across Clients', 
                 fontsize=13, fontweight='bold', y=0.995)
    
    strategies = ['FedAvg', 'FedProx', 'FedAdam']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Top row: C-Index variance
    for idx, n_clients in enumerate([5, 4, 3]):
        ax = axes[0, idx]
        
        for i, strategy in enumerate(strategies):
            data = results[n_clients][strategy]
            clients = data['clients']
            c_indices = data['c_index']
            stds = data['c_index_std']
            
            x = np.arange(len(clients))
            ax.errorbar(x, c_indices, yerr=stds, label=strategy, 
                       marker='o', capsize=5, linewidth=2, markersize=8,
                       color=colors[i], alpha=0.7)
        
        ax.set_title(f'{n_clients} Clients - C-Index Variability', fontweight='bold')
        ax.set_xlabel('Client')
        ax.set_ylabel('C-Index')
        ax.set_xticks(x)
        ax.set_xticklabels(clients)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.65, 0.95])
    
    # Bottom row: Coefficient of Variation (CV) comparison
    for idx, n_clients in enumerate([5, 4, 3]):
        ax = axes[1, idx]
        
        cv_data = []
        for strategy in strategies:
            data = results[n_clients][strategy]
            mean_cindex = np.mean(data['c_index'])
            std_cindex = np.std(data['c_index'])
            cv = (std_cindex / mean_cindex) * 100  # Coefficient of variation in %
            cv_data.append(cv)
        
        bars = ax.bar(strategies, cv_data, color=colors, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
        
        ax.set_title(f'{n_clients} Clients - Coefficient of Variation', fontweight='bold')
        ax.set_ylabel('CV (%) - Lower is More Stable')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'variance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'variance_analysis.png'}")
    plt.show()

# ========================================
# PLOT 6: Best Client Performance
# ========================================

def plot_best_client_performance():
    """Identify and visualize best performing client for each configuration"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    best_performances = []
    
    for n_clients in [3, 4, 5]:
        for strategy in ['FedAvg', 'FedProx', 'FedAdam']:
            data = results[n_clients][strategy]
            best_idx = np.argmax(data['c_index'])
            best_performances.append({
                'Configuration': f'{n_clients} Clients',
                'Strategy': strategy,
                'Best Client': data['clients'][best_idx],
                'C-Index': data['c_index'][best_idx],
                'C-Index Std': data['c_index_std'][best_idx],
                'AUC': data['auc'][best_idx],
                'IBS': data['ibs'][best_idx]
            })
    
    df_best = pd.DataFrame(best_performances)
    
    # Create grouped bar plot
    configs = df_best['Configuration'].unique()
    x = np.arange(len(configs))
    width = 0.25
    colors = {'FedAvg': '#3498db', 'FedProx': '#e74c3c', 'FedAdam': '#2ecc71'}
    
    for i, strategy in enumerate(['FedAvg', 'FedProx', 'FedAdam']):
        strategy_data = df_best[df_best['Strategy'] == strategy]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, strategy_data['C-Index'], width,
                     label=strategy, color=colors[strategy], alpha=0.8,
                     yerr=strategy_data['C-Index Std'], capsize=5)
        
        # Add client labels on bars
        for bar, (_, row) in zip(bars, strategy_data.iterrows()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{row['Best Client']}\n{row['C-Index']:.3f}",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Client C-Index', fontsize=13, fontweight='bold')
    ax.set_title('Best Performing Client in Each Configuration', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.7, 1.0])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'best_client_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'best_client_performance.png'}")
    plt.show()
    
    return df_best

# ========================================
# PLOT 7: Comprehensive Summary Table
# ========================================

def create_summary_table():
    """Create a comprehensive summary table of all results"""
    summary_data = []
    
    for n_clients in [5, 4, 3]:
        for strategy in ['FedAvg', 'FedProx', 'FedAdam']:
            data = results[n_clients][strategy]
            summary_data.append({
                'Clients': n_clients,
                'Strategy': strategy,
                'Rounds': data['rounds'],
                'Mean C-Index': f"{np.mean(data['c_index']):.3f} ± {np.std(data['c_index']):.3f}",
                'Mean AUC': f"{np.mean(data['auc']):.3f} ± {np.std(data['auc']):.3f}",
                'Mean IBS': f"{np.mean(data['ibs']):.3f} ± {np.std(data['ibs']):.3f}",
                'Best C-Index': f"{np.max(data['c_index']):.3f}",
                'Worst C-Index': f"{np.min(data['c_index']):.3f}"
            })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_summary.values,
                    colLabels=df_summary.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.08, 0.12, 0.08, 0.18, 0.18, 0.18, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(df_summary.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df_summary) + 1):
        for j in range(len(df_summary.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('Comprehensive Summary: DeepSurv Federated Learning Results', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_table.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'summary_table.png'}")
    plt.show()
    
    # Also save as CSV
    df_summary.to_csv(OUTPUT_DIR / 'summary_results.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'summary_results.csv'}")
    
    return df_summary

# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("="*60)
    print("DeepSurv Federated Learning Results Visualization")
    print("="*60)
    print()
    
    print("Generating visualizations...")
    print()
    
    # Generate all plots
    plot_cindex_comparison()
    plot_strategy_heatmaps()
    plot_average_performance()
    plot_convergence_rounds()
    plot_variance_analysis()
    df_best = plot_best_client_performance()
    df_summary = create_summary_table()
    
    print()
    print("="*60)
    print("All visualizations completed successfully!")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print("="*60)
    print()
    
    # Print key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    
    # Best overall configuration
    best_overall = df_best.loc[df_best['C-Index'].idxmax()]
    print(f"\n1. Best Overall Performance:")
    print(f"   - Configuration: {best_overall['Configuration']}")
    print(f"   - Strategy: {best_overall['Strategy']}")
    print(f"   - Client: {best_overall['Best Client']}")
    print(f"   - C-Index: {best_overall['C-Index']:.3f}")
    
    # Convergence efficiency
    print(f"\n2. Most Efficient Convergence:")
    fastest = df_summary.loc[df_summary['Rounds'].idxmin()]
    print(f"   - {fastest['Strategy']} with {fastest['Clients']} clients")
    print(f"   - Converges in {fastest['Rounds']} rounds")
    print(f"   - Mean C-Index: {fastest['Mean C-Index']}")
    
    # Most stable strategy
    print(f"\n3. Stability Analysis:")
    print(f"   - Check the variance analysis plots for coefficient of variation")
    print(f"   - Lower CV indicates more consistent performance across clients")
    
    print("\n" + "="*60)
