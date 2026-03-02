"""
Visualization script for DeepSurv Centralized and Local Training Results
Compares centralized and local approaches across different client configurations (3, 4, 5)
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
OUTPUT_DIR = Path("plots_output_Centralized_Local")
OUTPUT_DIR.mkdir(exist_ok=True)

# ========================================
# DATA PREPARATION
# ========================================

# Centralized results
centralized_results = {
    5: {
        'clients': ['C0', 'C1', 'C2', 'C3', 'C4'],
        'c_index': [0.666, 0.632, 0.627, 0.600, 0.705],
        'c_index_std': [0.179, 0.165, 0.162, 0.120, 0.234],
        'auc': [0.662, 0.640, 0.555, 0.653, 0.730],
        'auc_std': [0.167, 0.174, 0.080, 0.184, 0.255],
        'ibs': [0.170, 1.01, 0.154, 0.136, 0.040],
        'ibs_std': [0.028, 0.178, 0.006, 0.014, 0.007]
    },
    4: {
        'clients': ['C0', 'C1', 'C2', 'C3'],
        'c_index': [0.709, 0.636, 0.692, 0.675],
        'c_index_std': [0.168, 0.161, 0.158, 0.143],
        'auc': [0.690, 0.643, 0.636, 0.737],
        'auc_std': [0.153, 0.187, 0.124, 0.184],
        'ibs': [0.151, 0.836, 0.136, 0.108],
        'ibs_std': [0.021, 0.141, 0.013, 0.022]
    },
    3: {
        'clients': ['C0', 'C1', 'C2'],
        'c_index': [0.707, 0.598, 0.692],
        'c_index_std': [0.171, 0.096, 0.166],
        'auc': [0.698, 0.601, 0.617],
        'auc_std': [0.698, 0.601, 0.617],
        'ibs': [0.150, 0.864, 0.132],
        'ibs_std': [0.019, 0.176, 0.011]
    }
}

# Local results (all 5 clients trained independently)
local_results = {
    'clients': ['C0', 'C1', 'C2', 'C3', 'C4'],
    'c_index': [0.624, 0.505, 0.513, 0.538, 0.680],
    'c_index_std': [0.162, 0.213, 0.102, 0.171, 0.213],
    'auc': [0.614, 0.534, 0.528, 0.541, 0.717],
    'auc_std': [0.136, 0.184, 0.136, 0.224, 0.256],
    'ibs': [0.173, 0.294, 0.198, 0.103, 0.048],
    'ibs_std': [0.010, 0.037, 0.010, 0.013, 0.005]
}

# ========================================
# PLOT 1: Metric Comparison - Centralized vs Local (C-Index, AUC, IBS)
# ========================================

def plot_metric_comparison(metric='c_index', metric_name='C-Index', ylim=None):
    """Compare a specific metric between centralized and local training"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'{metric_name}: Centralized vs Local Training', 
                 fontsize=12, fontweight='bold', y=1.02)
    
    colors = {'Centralized': '#3498db', 'Local': '#e74c3c'}
    
    for idx, (n_clients, ax) in enumerate(zip([5, 4, 3], axes)):
        cent_data = centralized_results[n_clients]
        
        # Get matching clients for local data
        if n_clients == 5:
            local_clients = local_results['clients']
            local_metric = local_results[metric]
            local_std = local_results[f'{metric}_std']
        else:
            local_clients = local_results['clients'][:n_clients]
            local_metric = local_results[metric][:n_clients]
            local_std = local_results[f'{metric}_std'][:n_clients]
        
        x = np.arange(len(cent_data['clients']))
        width = 0.35
        
        # Centralized bars
        ax.bar(x - width/2, cent_data[metric], width, 
               label='Centralized', color=colors['Centralized'], alpha=0.8,
               yerr=cent_data[f'{metric}_std'], capsize=5)
        
        # Local bars
        ax.bar(x + width/2, local_metric, width, 
               label='Local', color=colors['Local'], alpha=0.8,
               yerr=local_std, capsize=5)
        
        ax.set_xlabel('Client', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{n_clients} Clients', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cent_data['clients'])
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        if ylim:
            ax.set_ylim(ylim)
    
    plt.tight_layout()
    filename = f'{metric}_centralized_vs_local.png'
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / filename}")
    plt.show()

def plot_cindex_comparison():
    """Compare C-Index between centralized and local training"""
    plot_metric_comparison(metric='c_index', metric_name='C-Index', ylim=[0.3, 1.0])

def plot_auc_comparison():
    """Compare AUC between centralized and local training"""
    plot_metric_comparison(metric='auc', metric_name='AUC', ylim=[0.3, 1.0])

def plot_ibs_comparison():
    """Compare IBS between centralized and local training"""
    plot_metric_comparison(metric='ibs', metric_name='IBS (lower is better)', ylim=[0, 1.2])

# ========================================
# PLOT 2: All Metrics Comparison Heatmap
# ========================================

def plot_metrics_heatmap():
    """Heatmap comparing all metrics for centralized and local"""
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    fig.suptitle('Metrics Heatmap: Centralized vs Local Training', 
                 fontsize=13, fontweight='bold', y=0.995)
    
    metrics = ['c_index', 'auc', 'ibs']
    metric_names = ['C-Index', 'AUC', 'IBS']
    training_modes = ['Centralized', 'Local']
    
    # Use 5 clients configuration for complete comparison
    n_clients = 5
    cent_data = centralized_results[n_clients]
    
    for row_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        # Centralized column
        ax_cent = axes[row_idx, 0]
        cent_values = np.array(cent_data[metric]).reshape(1, -1)
        
        if metric == 'ibs':
            cmap = 'RdYlGn_r'
            vmin, vmax = 0, 0.3
        else:
            cmap = 'RdYlGn'
            vmin, vmax = 0.4, 1.0
        
        im_cent = ax_cent.imshow(cent_values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax_cent.set_xticks(np.arange(len(cent_data['clients'])))
        ax_cent.set_xticklabels(cent_data['clients'])
        ax_cent.set_yticks([0])
        ax_cent.set_yticklabels(['Centralized'])
        
        # Add values
        for j in range(len(cent_data['clients'])):
            ax_cent.text(j, 0, f'{cent_values[0, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=9)
        
        if row_idx == 0:
            ax_cent.set_title('Centralized (5 clients)', fontsize=11, fontweight='bold')
        if row_idx == 2:
            ax_cent.set_xlabel('Client ID', fontsize=10)
        
        ax_cent.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        
        # Local column
        ax_local = axes[row_idx, 1]
        local_values = np.array(local_results[metric]).reshape(1, -1)
        
        im_local = ax_local.imshow(local_values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax_local.set_xticks(np.arange(len(local_results['clients'])))
        ax_local.set_xticklabels(local_results['clients'])
        ax_local.set_yticks([0])
        ax_local.set_yticklabels(['Local'])
        
        # Add values
        for j in range(len(local_results['clients'])):
            ax_local.text(j, 0, f'{local_values[0, j]:.3f}',
                         ha="center", va="center", color="black", fontsize=9)
        
        if row_idx == 0:
            ax_local.set_title('Local (5 clients)', fontsize=11, fontweight='bold')
        if row_idx == 2:
            ax_local.set_xlabel('Client ID', fontsize=10)
        
        # Colorbar
        plt.colorbar(im_local, ax=ax_local, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'metrics_heatmap.png'}")
    plt.show()

# ========================================
# PLOT 3: Average Performance Comparison
# ========================================

def plot_average_performance():
    """Compare average metrics across all approaches"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Average Performance: Centralized vs Local', 
                 fontsize=12, fontweight='bold', y=1.02)
    
    metrics = ['c_index', 'auc', 'ibs']
    metric_names = ['C-Index', 'AUC', 'IBS (lower is better)']
    colors = {'Centralized (5)': '#3498db', 'Centralized (4)': '#5dade2', 
              'Centralized (3)': '#85c1e9', 'Local': '#e74c3c'}
    
    for ax, metric, metric_name in zip(axes, metrics, metric_names):
        approaches = []
        means = []
        stds = []
        
        # Centralized results
        for n_clients in [5, 4, 3]:
            data = centralized_results[n_clients]
            approaches.append(f'Cent ({n_clients}C)')
            means.append(np.mean(data[metric]))
            # Propagate uncertainty
            stds.append(np.sqrt(np.sum(np.array(data[f'{metric}_std'])**2)) / len(data[metric]))
        
        # Local results (5 clients)
        approaches.append('Local (5C)')
        means.append(np.mean(local_results[metric]))
        stds.append(np.sqrt(np.sum(np.array(local_results[f'{metric}_std'])**2)) / len(local_results[metric]))
        
        x_pos = np.arange(len(approaches))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8,
                     color=['#3498db', '#5dade2', '#85c1e9', '#e74c3c'])
        
        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(approaches, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        if metric != 'ibs':
            ax.set_ylim([0.4, 0.9])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'average_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'average_performance_comparison.png'}")
    plt.show()

# ========================================
# PLOT 4: Variance/Stability Analysis
# ========================================

def plot_variance_comparison():
    """Compare variance between centralized and local approaches"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Performance Stability: Centralized vs Local', 
                 fontsize=13, fontweight='bold', y=0.995)
    
    # Top row: C-Index with error bars for 5 clients
    ax1 = axes[0, 0]
    n_clients = 5
    cent_data = centralized_results[n_clients]
    x = np.arange(len(cent_data['clients']))
    
    ax1.errorbar(x, cent_data['c_index'], yerr=cent_data['c_index_std'],
                 marker='o', capsize=5, linewidth=2, markersize=8,
                 color='#3498db', alpha=0.7, label='Centralized')
    ax1.errorbar(x, local_results['c_index'], yerr=local_results['c_index_std'],
                 marker='s', capsize=5, linewidth=2, markersize=8,
                 color='#e74c3c', alpha=0.7, label='Local')
    
    ax1.set_title('C-Index Variability (5 Clients)', fontweight='bold')
    ax1.set_xlabel('Client')
    ax1.set_ylabel('C-Index')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cent_data['clients'])
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.3, 1.0])
    
    # Top right: Coefficient of Variation comparison
    ax2 = axes[0, 1]
    approaches = []
    cv_values = []
    
    for n_clients in [5, 4, 3]:
        data = centralized_results[n_clients]
        mean_cindex = np.mean(data['c_index'])
        std_cindex = np.std(data['c_index'])
        cv = (std_cindex / mean_cindex) * 100
        approaches.append(f'Cent ({n_clients}C)')
        cv_values.append(cv)
    
    # Local CV
    mean_local = np.mean(local_results['c_index'])
    std_local = np.std(local_results['c_index'])
    cv_local = (std_local / mean_local) * 100
    approaches.append('Local (5C)')
    cv_values.append(cv_local)
    
    bars = ax2.bar(approaches, cv_values, alpha=0.8,
                  color=['#3498db', '#5dade2', '#85c1e9', '#e74c3c'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    ax2.set_title('Coefficient of Variation (C-Index)', fontweight='bold')
    ax2.set_ylabel('CV (%) - Lower is More Stable')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Bottom left: Standard deviation comparison
    ax3 = axes[1, 0]
    x_pos = np.arange(len(approaches))
    
    std_values = []
    for n_clients in [5, 4, 3]:
        data = centralized_results[n_clients]
        std_values.append(np.std(data['c_index']))
    std_values.append(np.std(local_results['c_index']))
    
    bars = ax3.bar(x_pos, std_values, alpha=0.8,
                  color=['#3498db', '#5dade2', '#85c1e9', '#e74c3c'])
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax3.set_title('Standard Deviation (C-Index)', fontweight='bold')
    ax3.set_ylabel('Std Dev - Lower is More Stable')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(approaches, rotation=15, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Bottom right: IBS comparison (5 clients)
    ax4 = axes[1, 1]
    x = np.arange(5)
    width = 0.35
    
    cent_ibs = centralized_results[5]['ibs']
    cent_ibs_std = centralized_results[5]['ibs_std']
    local_ibs = local_results['ibs']
    local_ibs_std = local_results['ibs_std']
    
    ax4.bar(x - width/2, cent_ibs, width, label='Centralized',
            color='#3498db', alpha=0.8, yerr=cent_ibs_std, capsize=5)
    ax4.bar(x + width/2, local_ibs, width, label='Local',
            color='#e74c3c', alpha=0.8, yerr=local_ibs_std, capsize=5)
    
    ax4.set_title('IBS Comparison (5 Clients)', fontweight='bold')
    ax4.set_xlabel('Client')
    ax4.set_ylabel('IBS (lower is better)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(centralized_results[5]['clients'])
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'variance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'variance_comparison.png'}")
    plt.show()

# ========================================
# PLOT 5: Summary Table
# ========================================

def create_summary_table():
    """Create comprehensive summary table"""
    summary_data = []
    
    # Centralized results
    for n_clients in [5, 4, 3]:
        data = centralized_results[n_clients]
        summary_data.append({
            'Training Mode': f'Centralized',
            'Clients': n_clients,
            'Mean C-Index': f"{np.mean(data['c_index']):.3f} ± {np.std(data['c_index']):.3f}",
            'Mean AUC': f"{np.mean(data['auc']):.3f} ± {np.std(data['auc']):.3f}",
            'Mean IBS': f"{np.mean(data['ibs']):.3f} ± {np.std(data['ibs']):.3f}",
            'Best C-Index': f"{np.max(data['c_index']):.3f}",
            'Worst C-Index': f"{np.min(data['c_index']):.3f}",
            'CV (%)': f"{(np.std(data['c_index']) / np.mean(data['c_index']) * 100):.2f}"
        })
    
    # Local results
    summary_data.append({
        'Training Mode': 'Local',
        'Clients': 5,
        'Mean C-Index': f"{np.mean(local_results['c_index']):.3f} ± {np.std(local_results['c_index']):.3f}",
        'Mean AUC': f"{np.mean(local_results['auc']):.3f} ± {np.std(local_results['auc']):.3f}",
        'Mean IBS': f"{np.mean(local_results['ibs']):.3f} ± {np.std(local_results['ibs']):.3f}",
        'Best C-Index': f"{np.max(local_results['c_index']):.3f}",
        'Worst C-Index': f"{np.min(local_results['c_index']):.3f}",
        'CV (%)': f"{(np.std(local_results['c_index']) / np.mean(local_results['c_index']) * 100):.2f}"
    })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_summary.values,
                    colLabels=df_summary.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12, 0.08, 0.15, 0.15, 0.15, 0.12, 0.12, 0.08])
    
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
            
            # Highlight local row
            if df_summary.iloc[i-1]['Training Mode'] == 'Local':
                table[(i, j)].set_facecolor('#fadbd8')
    
    plt.title('Summary: Centralized vs Local Training Results', 
             fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_table.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'summary_table.png'}")
    plt.show()
    
    # Save CSV
    df_summary.to_csv(OUTPUT_DIR / 'summary_centralized_local.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'summary_centralized_local.csv'}")
    
    return df_summary

# ========================================
# PLOT 6: Client-Specific Performance
# ========================================

def plot_client_specific_analysis():
    """Analyze individual client performance across training modes"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Use 5 clients for complete comparison
    clients = centralized_results[5]['clients']
    x = np.arange(len(clients))
    width = 0.25
    
    # Centralized
    cent_cindex = centralized_results[5]['c_index']
    cent_std = centralized_results[5]['c_index_std']
    
    # Local
    local_cindex = local_results['c_index']
    local_std = local_results['c_index_std']
    
    bars1 = ax.bar(x - width, cent_cindex, width, label='Centralized',
                   color='#3498db', alpha=0.8, yerr=cent_std, capsize=5)
    bars2 = ax.bar(x, local_cindex, width, label='Local',
                   color='#e74c3c', alpha=0.8, yerr=local_std, capsize=5)
    
    # Calculate and show improvement/degradation
    for i, (cent, local) in enumerate(zip(cent_cindex, local_cindex)):
        diff = cent - local
        color = 'green' if diff > 0 else 'red'
        ax.text(x[i] + width, max(cent, local) + 0.05,
               f'{diff:+.3f}', ha='center', va='bottom',
               color=color, fontweight='bold', fontsize=8)
    
    ax.set_xlabel('Client', fontsize=12, fontweight='bold')
    ax.set_ylabel('C-Index', fontsize=12, fontweight='bold')
    ax.set_title('Client-Specific C-Index: Centralized vs Local Training\n(Centralized - Local difference shown above bars)',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(clients)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.3, 1.0])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'client_specific_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'client_specific_analysis.png'}")
    plt.show()

# ========================================
# PLOT 7: Individual Client Metric Comparison (All Metrics)
# ========================================

def plot_individual_client_all_metrics():
    """Show all metrics for each individual client comparing centralized vs local"""
    clients = centralized_results[5]['clients']
    metrics = ['c_index', 'auc', 'ibs']
    metric_names = ['C-Index', 'AUC', 'IBS']
    
    fig, axes = plt.subplots(len(clients), len(metrics), figsize=(15, 12))
    fig.suptitle('Individual Client Performance: Centralized vs Local (All Metrics)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    for client_idx, client in enumerate(clients):
        for metric_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[client_idx, metric_idx]
            
            # Get values
            cent_val = centralized_results[5][metric][client_idx]
            cent_std = centralized_results[5][f'{metric}_std'][client_idx]
            local_val = local_results[metric][client_idx]
            local_std = local_results[f'{metric}_std'][client_idx]
            
            # Bar plot
            x_pos = [0, 1]
            values = [cent_val, local_val]
            errors = [cent_std, local_std]
            colors = ['#3498db', '#e74c3c']
            labels = ['Centralized', 'Local']
            
            bars = ax.bar(x_pos, values, color=colors, alpha=0.8, 
                         yerr=errors, capsize=5, width=0.6)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Calculate difference
            diff = cent_val - local_val
            if metric == 'ibs':
                # For IBS, lower is better, so negative diff is good
                improvement = -diff
            else:
                improvement = diff
            
            improvement_pct = (improvement / local_val) * 100 if local_val != 0 else 0
            
            # Title with client name and improvement
            if metric_idx == 1:  # Middle column
                title = f'{client}\n{metric_name}\nImpr: {improvement_pct:+.1f}%'
            else:
                title = f'{metric_name}\nImpr: {improvement_pct:+.1f}%'
            
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set appropriate y-axis limits
            if metric == 'ibs':
                ax.set_ylim([0, max(values) * 1.3])
            else:
                ax.set_ylim([0.3, 1.0])
            
            # Only show y-label on leftmost column
            if metric_idx == 0:
                ax.set_ylabel(client, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'individual_client_all_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'individual_client_all_metrics.png'}")
    plt.show()

# ========================================
# PLOT 8: Spider/Radar Plot for Each Client
# ========================================

def plot_client_radar_charts():
    """Create radar charts showing performance profile for each client"""
    from math import pi
    
    clients = centralized_results[5]['clients']
    n_clients = len(clients)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(projection='polar'))
    fig.suptitle('Individual Client Performance Profiles: Centralized vs Local', 
                 fontsize=14, fontweight='bold', y=0.995)
    axes = axes.flatten()
    
    # Metrics to plot (normalize IBS to 0-1 scale where 1 is best)
    categories = ['C-Index', 'AUC', 'IBS\n(inverted)']
    num_vars = len(categories)
    
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    for idx, (client, ax) in enumerate(zip(clients, axes[:n_clients])):
        # Get metrics
        cent_cindex = centralized_results[5]['c_index'][idx]
        cent_auc = centralized_results[5]['auc'][idx]
        cent_ibs = centralized_results[5]['ibs'][idx]
        
        local_cindex = local_results['c_index'][idx]
        local_auc = local_results['auc'][idx]
        local_ibs = local_results['ibs'][idx]
        
        # Normalize IBS (invert so higher is better, and scale to 0-1)
        # IBS typically ranges from 0 to ~1, so we'll use 1 - normalized_ibs
        max_ibs = 1.0  # assumed max for scaling
        cent_ibs_norm = 1 - min(cent_ibs / max_ibs, 1.0)
        local_ibs_norm = 1 - min(local_ibs / max_ibs, 1.0)
        
        # Create data
        cent_values = [cent_cindex, cent_auc, cent_ibs_norm]
        local_values = [local_cindex, local_auc, local_ibs_norm]
        
        cent_values += cent_values[:1]
        local_values += local_values[:1]
        
        # Plot
        ax.plot(angles, cent_values, 'o-', linewidth=2, label='Centralized', 
               color='#3498db', markersize=8)
        ax.fill(angles, cent_values, alpha=0.15, color='#3498db')
        
        ax.plot(angles, local_values, 'o-', linewidth=2, label='Local', 
               color='#e74c3c', markersize=8)
        ax.fill(angles, local_values, alpha=0.15, color='#e74c3c')
        
        # Fix axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.grid(True)
        
        ax.set_title(f'{client}', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'client_radar_charts.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'client_radar_charts.png'}")
    plt.show()

# ========================================
# PLOT 9: Per-Client Improvement Analysis
# ========================================

def plot_client_improvement_analysis():
    """Analyze improvement/degradation for each client across all metrics"""
    clients = centralized_results[5]['clients']
    metrics = ['c_index', 'auc', 'ibs']
    metric_names = ['C-Index', 'AUC', 'IBS']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Per-Client Improvement: Centralized vs Local\n(Positive = Centralized Better)', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    for ax, metric, metric_name in zip(axes, metrics, metric_names):
        improvements = []
        
        for idx in range(len(clients)):
            cent_val = centralized_results[5][metric][idx]
            local_val = local_results[metric][idx]
            
            # For IBS, lower is better, so we invert the difference
            if metric == 'ibs':
                improvement = local_val - cent_val  # Positive means centralized is better (lower)
            else:
                improvement = cent_val - local_val  # Positive means centralized is better (higher)
            
            improvements.append(improvement)
        
        # Create bar plot
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.bar(range(len(clients)), improvements, color=colors, alpha=0.7)
        
        # Add value labels
        for idx, (bar, imp) in enumerate(zip(bars, improvements)):
            height = bar.get_height()
            y_pos = height + 0.01 if height > 0 else height - 0.01
            va = 'bottom' if height > 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{imp:+.3f}', ha='center', va=va, fontsize=9, fontweight='bold')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Client', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{metric_name} Improvement', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(clients)))
        ax.set_xticklabels(clients)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'client_improvement_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'client_improvement_analysis.png'}")
    plt.show()

# ========================================
# PLOT 10: Centralized Performance Across Different Client Configurations
# ========================================

def plot_centralized_per_client_across_configs():
    """Show how each client performs in centralized training with 5, 4, and 3 clients"""
    metrics = ['c_index', 'auc', 'ibs']
    metric_names = ['C-Index', 'AUC', 'IBS']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Centralized Performance Per Client: 5 vs 4 vs 3 Clients Configuration', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    # All clients that appear in different configs
    all_clients = ['C0', 'C1', 'C2', 'C3', 'C4']
    
    for ax, metric, metric_name in zip(axes, metrics, metric_names):
        x_pos = np.arange(len(all_clients))
        width = 0.25
        
        values_5c = []
        values_4c = []
        values_3c = []
        stds_5c = []
        stds_4c = []
        stds_3c = []
        
        for client in all_clients:
            # 5 clients config
            if client in centralized_results[5]['clients']:
                idx = centralized_results[5]['clients'].index(client)
                values_5c.append(centralized_results[5][metric][idx])
                stds_5c.append(centralized_results[5][f'{metric}_std'][idx])
            else:
                values_5c.append(None)
                stds_5c.append(None)
            
            # 4 clients config
            if client in centralized_results[4]['clients']:
                idx = centralized_results[4]['clients'].index(client)
                values_4c.append(centralized_results[4][metric][idx])
                stds_4c.append(centralized_results[4][f'{metric}_std'][idx])
            else:
                values_4c.append(None)
                stds_4c.append(None)
            
            # 3 clients config
            if client in centralized_results[3]['clients']:
                idx = centralized_results[3]['clients'].index(client)
                values_3c.append(centralized_results[3][metric][idx])
                stds_3c.append(centralized_results[3][f'{metric}_std'][idx])
            else:
                values_3c.append(None)
                stds_3c.append(None)
        
        # Plot bars only where data exists
        for i in range(len(all_clients)):
            if values_5c[i] is not None:
                ax.bar(i - width, values_5c[i], width, color='#3498db', alpha=0.8,
                      yerr=stds_5c[i], capsize=4, label='5 Clients' if i == 0 else '')
            if values_4c[i] is not None:
                ax.bar(i, values_4c[i], width, color='#5dade2', alpha=0.8,
                      yerr=stds_4c[i], capsize=4, label='4 Clients' if i == 0 else '')
            if values_3c[i] is not None:
                ax.bar(i + width, values_3c[i], width, color='#85c1e9', alpha=0.8,
                      yerr=stds_3c[i], capsize=4, label='3 Clients' if i == 0 else '')
        
        # Add value labels
        for i in range(len(all_clients)):
            if values_5c[i] is not None:
                ax.text(i - width, values_5c[i] + (stds_5c[i] if stds_5c[i] else 0) + 0.02,
                       f'{values_5c[i]:.3f}', ha='center', va='bottom', fontsize=7, rotation=0)
            if values_4c[i] is not None:
                ax.text(i, values_4c[i] + (stds_4c[i] if stds_4c[i] else 0) + 0.02,
                       f'{values_4c[i]:.3f}', ha='center', va='bottom', fontsize=7, rotation=0)
            if values_3c[i] is not None:
                ax.text(i + width, values_3c[i] + (stds_3c[i] if stds_3c[i] else 0) + 0.02,
                       f'{values_3c[i]:.3f}', ha='center', va='bottom', fontsize=7, rotation=0)
        
        ax.set_xlabel('Client', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_clients)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set appropriate y-axis limits
        if metric == 'ibs':
            ax.set_ylim([0, 1.2])
        else:
            ax.set_ylim([0.4, 0.9])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'centralized_per_client_across_configs.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'centralized_per_client_across_configs.png'}")
    plt.show()

# ========================================
# PLOT 11: Detailed Client Performance Panels
# ========================================

def plot_detailed_client_panels():
    """Create detailed performance panel for each client showing all aspects"""
    clients = centralized_results[5]['clients']
    
    for client_idx, client in enumerate(clients):
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(f'Detailed Performance Analysis: {client}', 
                     fontsize=14, fontweight='bold')
        
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: All metrics comparison
        ax1 = fig.add_subplot(gs[0, :])
        metrics = ['c_index', 'auc', 'ibs']
        metric_names = ['C-Index', 'AUC', 'IBS']
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        cent_vals = [centralized_results[5][m][client_idx] for m in metrics]
        cent_stds = [centralized_results[5][f'{m}_std'][client_idx] for m in metrics]
        local_vals = [local_results[m][client_idx] for m in metrics]
        local_stds = [local_results[f'{m}_std'][client_idx] for m in metrics]
        
        ax1.bar(x_pos - width/2, cent_vals, width, label='Centralized',
               color='#3498db', alpha=0.8, yerr=cent_stds, capsize=5)
        ax1.bar(x_pos + width/2, local_vals, width, label='Local',
               color='#e74c3c', alpha=0.8, yerr=local_stds, capsize=5)
        
        # Add values on bars
        for i, (cv, lv) in enumerate(zip(cent_vals, local_vals)):
            ax1.text(i - width/2, cv + cent_stds[i] + 0.02, f'{cv:.3f}',
                    ha='center', va='bottom', fontsize=9)
            ax1.text(i + width/2, lv + local_stds[i] + 0.02, f'{lv:.3f}',
                    ha='center', va='bottom', fontsize=9)
        
        ax1.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
        ax1.set_title('All Metrics Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(metric_names)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Panel 2: Absolute improvement
        ax2 = fig.add_subplot(gs[1, 0])
        improvements = []
        for m in metrics:
            cent_val = centralized_results[5][m][client_idx]
            local_val = local_results[m][client_idx]
            if m == 'ibs':
                imp = local_val - cent_val
            else:
                imp = cent_val - local_val
            improvements.append(imp)
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax2.bar(metric_names, improvements, color=colors, alpha=0.7)
        
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            y_pos = height + 0.005 if height > 0 else height - 0.005
            va = 'bottom' if height > 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{imp:+.3f}', ha='center', va=va, fontsize=9, fontweight='bold')
        
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel('Absolute Improvement', fontsize=10, fontweight='bold')
        ax2.set_title('Centralized Improvement', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Percentage improvement
        ax3 = fig.add_subplot(gs[1, 1])
        pct_improvements = []
        for m, local_val in zip(metrics, local_vals):
            cent_val = centralized_results[5][m][client_idx]
            if m == 'ibs':
                imp = ((local_val - cent_val) / local_val * 100) if local_val != 0 else 0
            else:
                imp = ((cent_val - local_val) / local_val * 100) if local_val != 0 else 0
            pct_improvements.append(imp)
        
        colors = ['green' if imp > 0 else 'red' for imp in pct_improvements]
        bars = ax3.bar(metric_names, pct_improvements, color=colors, alpha=0.7)
        
        for bar, imp in zip(bars, pct_improvements):
            height = bar.get_height()
            y_pos = height + 1 if height > 0 else height - 1
            va = 'bottom' if height > 0 else 'top'
            ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{imp:+.1f}%', ha='center', va=va, fontsize=9, fontweight='bold')
        
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.set_ylabel('% Improvement', fontsize=10, fontweight='bold')
        ax3.set_title('Relative Improvement', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Panel 4: Summary stats
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        
        summary_text = f"""CLIENT {client} SUMMARY
        
C-Index:
  Centralized: {cent_vals[0]:.3f} ± {cent_stds[0]:.3f}
  Local: {local_vals[0]:.3f} ± {local_stds[0]:.3f}
  Difference: {improvements[0]:+.3f} ({pct_improvements[0]:+.1f}%)

AUC:
  Centralized: {cent_vals[1]:.3f} ± {cent_stds[1]:.3f}
  Local: {local_vals[1]:.3f} ± {local_stds[1]:.3f}
  Difference: {improvements[1]:+.3f} ({pct_improvements[1]:+.1f}%)

IBS:
  Centralized: {cent_vals[2]:.3f} ± {cent_stds[2]:.3f}
  Local: {local_vals[2]:.3f} ± {local_stds[2]:.3f}
  Difference: {improvements[2]:+.3f} ({pct_improvements[2]:+.1f}%)

OVERALL:
  {'Centralized performs BETTER' if sum(imp > 0 for imp in improvements) >= 2 else 'Local performs BETTER'}
  {sum(imp > 0 for imp in improvements)}/3 metrics improved
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.savefig(OUTPUT_DIR / f'detailed_panel_{client}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR / f'detailed_panel_{client}.png'}")
        plt.show()

# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("="*60)
    print("DeepSurv Centralized vs Local Training Visualization")
    print("="*60)
    print()
    
    print("Generating visualizations...")
    print()
    
    # Generate all plots
    plot_cindex_comparison()
    plot_auc_comparison()
    plot_ibs_comparison()
    plot_metrics_heatmap()
    plot_average_performance()
    plot_variance_comparison()
    plot_client_specific_analysis()
    df_summary = create_summary_table()
    
    # NEW: Generate individual client-focused plots
    print("\nGenerating individual client analysis plots...")
    plot_individual_client_all_metrics()
    plot_client_radar_charts()
    plot_client_improvement_analysis()
    plot_centralized_per_client_across_configs()
    plot_detailed_client_panels()
    
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
    
    # Average performance comparison
    cent_5_mean = np.mean(centralized_results[5]['c_index'])
    local_mean = np.mean(local_results['c_index'])
    
    print(f"\n1. Average C-Index Comparison (5 clients):")
    print(f"   - Centralized: {cent_5_mean:.3f}")
    print(f"   - Local: {local_mean:.3f}")
    print(f"   - Difference: {cent_5_mean - local_mean:+.3f} (Centralized {'better' if cent_5_mean > local_mean else 'worse'})")
    
    # Stability comparison
    cent_cv = (np.std(centralized_results[5]['c_index']) / cent_5_mean) * 100
    local_cv = (np.std(local_results['c_index']) / local_mean) * 100
    
    print(f"\n2. Stability (Coefficient of Variation):")
    print(f"   - Centralized: {cent_cv:.2f}%")
    print(f"   - Local: {local_cv:.2f}%")
    print(f"   - {'Centralized' if cent_cv < local_cv else 'Local'} is more stable")
    
    # Best client
    best_cent_idx = np.argmax(centralized_results[5]['c_index'])
    best_local_idx = np.argmax(local_results['c_index'])
    
    print(f"\n3. Best Performing Client:")
    print(f"   - Centralized: {centralized_results[5]['clients'][best_cent_idx]} "
          f"(C-Index: {centralized_results[5]['c_index'][best_cent_idx]:.3f})")
    print(f"   - Local: {local_results['clients'][best_local_idx]} "
          f"(C-Index: {local_results['c_index'][best_local_idx]:.3f})")
    
    # Problematic clients (highlighting C1's unusual IBS values)
    print(f"\n4. Outlier Detection:")
    print(f"   - Client C1 shows anomalously high IBS values:")
    print(f"     * Centralized (5C): {centralized_results[5]['ibs'][1]:.3f}")
    print(f"     * Centralized (4C): {centralized_results[4]['ibs'][1]:.3f}")
    print(f"     * Centralized (3C): {centralized_results[3]['ibs'][1]:.3f}")
    print(f"     * Local: {local_results['ibs'][1]:.3f}")
    print(f"   - This suggests potential data quality or preprocessing issues for C1")
    
    print("\n" + "="*60)
