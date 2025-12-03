"""
Visualization module for plotting unlearning results.

Provides functions for creating plots to visualize accuracy comparisons,
Pareto frontiers, feature activations, and hyperparameter sweeps.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

from ..features.identifier import FeatureIdentifier

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_accuracy_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str = "accuracy_comparison.png",
    figsize: tuple = (10, 6)
):
    """
    Plot accuracy comparison across methods.
    
    Args:
        results: Dict mapping method names to dataset accuracies
                 e.g., {'Baseline': {'WMDP-Bio': 0.58, 'MMLU': 0.65}, ...}
        save_path: Path to save the plot
        figsize: Figure size
    """
    methods = list(results.keys())
    datasets = list(results[methods[0]].keys())
    
    x = np.arange(len(datasets))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(methods):
        values = [results[method][ds] for ds in datasets]
        ax.bar(x + i * width, values, width, label=method, color=colors[i])
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison Across Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_pareto_frontier(
    results_list: List[Dict],
    save_path: str = "pareto_frontier.png",
    figsize: tuple = (10, 8)
):
    """
    Plot Pareto frontier: WMDP accuracy vs MMLU accuracy.
    
    Args:
        results_list: List of dicts with 'method', 'wmdp', 'mmlu' keys
                      e.g., [{'method': 'Baseline', 'wmdp': 0.58, 'mmlu': 0.65}, ...]
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color and marker mapping
    method_styles = {
        'Baseline': ('red', 'o'),
        'Clamp Prime': ('blue', 's'),
        'Refusal Clamp': ('green', '^'),
        'RMU': ('orange', 'D')
    }
    
    plotted_labels = set()
    
    for result in results_list:
        method = result['method']
        
        # Find matching style
        style_key = None
        for key in method_styles:
            if key in method:
                style_key = key
                break
        
        if style_key:
            color, marker = method_styles[style_key]
        else:
            color, marker = 'gray', 'o'
        
        # Only add label once per method type
        label = method if method not in plotted_labels else None
        plotted_labels.add(method)
        
        ax.scatter(
            result['wmdp'], result['mmlu'],
            c=color, marker=marker, s=100, alpha=0.7,
            label=label
        )
        
        # Add text annotation
        ax.annotate(
            result['method'],
            (result['wmdp'], result['mmlu']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, alpha=0.7
        )
    
    # Draw reference lines
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.3, label='Random guess')
    ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('WMDP-Bio Accuracy (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MMLU Accuracy (Higher is Better)', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Frontier: Forgetting vs Retention', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_feature_activation_heatmap(
    forget_latents: torch.Tensor,
    retain_latents: torch.Tensor,
    harmful_features: List[int],
    save_path: str = "feature_heatmap.png",
    top_n: int = 50,
    figsize: tuple = (14, 4)
):
    """
    Heatmap showing activation frequency of top harmful features
    on forget vs retain datasets.
    
    Args:
        forget_latents: Latent activations on forget dataset
        retain_latents: Latent activations on retain dataset
        harmful_features: List of identified harmful feature indices
        save_path: Path to save the plot
        top_n: Number of features to show
        figsize: Figure size
    """
    # Compute frequencies
    forget_freq = FeatureIdentifier.compute_activation_frequency(forget_latents, threshold=0.01)
    retain_freq = FeatureIdentifier.compute_activation_frequency(retain_latents, threshold=0.01)
    
    # Get top harmful features
    features_to_plot = harmful_features[:top_n]
    
    if len(features_to_plot) == 0:
        print("No features to plot")
        return
    
    # Create matrix [2, top_n]
    matrix = torch.stack([
        forget_freq[features_to_plot],
        retain_freq[features_to_plot]
    ]).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Forget Dataset', 'Retain Dataset'])
    ax.set_xlabel('Feature Index', fontsize=12, fontweight='bold')
    ax.set_title(f'Activation Frequency of Top {len(features_to_plot)} Harmful Features',
                 fontsize=14, fontweight='bold')
    
    # Add feature indices as x-tick labels
    if len(features_to_plot) <= 30:
        ax.set_xticks(range(len(features_to_plot)))
        ax.set_xticklabels(features_to_plot, rotation=45, ha='right', fontsize=8)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Activation Frequency', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_hyperparameter_sweep(
    sweep_results: List[Dict],
    param_name: str,
    save_path: str = "hyperparam_sweep.png",
    figsize: tuple = (14, 5)
):
    """
    Plot effect of hyperparameter on performance.
    
    Args:
        sweep_results: List of dicts with 'param_value', 'wmdp_acc', 'mmlu_acc', 'alignment'
        param_name: Name of the parameter being swept
        save_path: Path to save the plot
        figsize: Figure size
    """
    param_values = [r['param_value'] for r in sweep_results]
    wmdp_accs = [r['wmdp_acc'] for r in sweep_results]
    mmlu_accs = [r['mmlu_acc'] for r in sweep_results]
    alignments = [r['alignment'] for r in sweep_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Accuracies
    ax1.plot(param_values, wmdp_accs, 'o-', label='WMDP-Bio (lower is better)',
             linewidth=2, markersize=8, color='red')
    ax1.plot(param_values, mmlu_accs, 's-', label='MMLU (higher is better)',
             linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel(param_name, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title(f'Effect of {param_name} on Accuracy', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Alignment
    ax2.plot(param_values, alignments, 'D-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel(param_name, fontsize=12, fontweight='bold')
    ax2.set_ylabel('Alignment Score', fontsize=12, fontweight='bold')
    ax2.set_title(f'Effect of {param_name} on Alignment', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_sae_reconstruction_quality(
    sae_wrapper,
    activations: torch.Tensor,
    save_path: str = "sae_reconstruction.png",
    sample_size: int = 1000,
    figsize: tuple = (14, 5)
):
    """
    Plot SAE reconstruction quality metrics.
    
    Args:
        sae_wrapper: GemmaScopeWrapper or similar with forward() method
        activations: Input activations to test reconstruction
        save_path: Path to save the plot
        sample_size: Number of samples to use
        figsize: Figure size
    """
    sae_wrapper.sae.eval()
    
    with torch.no_grad():
        # Flatten if needed
        if activations.dim() == 3:
            B, S, D = activations.shape
            acts_flat = activations.reshape(B * S, D)
        else:
            acts_flat = activations
        
        # Sample subset
        sample_size = min(sample_size, acts_flat.shape[0])
        indices = torch.randperm(acts_flat.shape[0])[:sample_size]
        acts_sample = acts_flat[indices]
        
        print("Computing SAE reconstruction...")
        recon, latents = sae_wrapper.forward(acts_sample)
        
        recon = recon.cpu()
        acts_sample = acts_sample.cpu()
        
        # Compute MSE per sample
        mse_per_sample = ((recon - acts_sample) ** 2).mean(dim=1)
        
        # Compute sparsity
        sparsity = (latents.abs() > 0.01).float().mean(dim=1).cpu()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Reconstruction MSE
    ax1.hist(mse_per_sample.numpy(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Reconstruction MSE', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'SAE Reconstruction Quality\n(Mean MSE: {mse_per_sample.mean():.4f})',
                  fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Sparsity
    ax2.hist(sparsity.numpy(), bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel('Fraction of Active Features', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'SAE Sparsity\n(Mean: {sparsity.mean():.3f})',
                  fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_feature_frequency_scatter(
    forget_latents: torch.Tensor,
    retain_latents: torch.Tensor,
    harmful_features: List[int],
    threshold: float = 0.01,
    save_path: str = "feature_scatter.png",
    figsize: tuple = (10, 10)
):
    """
    Scatter plot of feature frequencies on forget vs retain datasets.
    
    Args:
        forget_latents: Latent activations on forget dataset
        retain_latents: Latent activations on retain dataset
        harmful_features: List of identified harmful feature indices
        threshold: Activation threshold
        save_path: Path to save the plot
        figsize: Figure size
    """
    forget_freq = FeatureIdentifier.compute_activation_frequency(forget_latents, threshold).cpu().numpy()
    retain_freq = FeatureIdentifier.compute_activation_frequency(retain_latents, threshold).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all features
    ax.scatter(retain_freq, forget_freq, alpha=0.3, s=10, label='All features')
    
    # Highlight harmful features
    if harmful_features:
        ax.scatter(
            retain_freq[harmful_features],
            forget_freq[harmful_features],
            color='red', s=50, alpha=0.8,
            label=f'Harmful features (n={len(harmful_features)})'
        )
    
    # Diagonal line
    ax.plot([0, max(retain_freq.max(), forget_freq.max())],
            [0, max(retain_freq.max(), forget_freq.max())],
            'k--', alpha=0.3, label='y=x')
    
    ax.set_xlabel('Retain Dataset Frequency', fontsize=12, fontweight='bold')
    ax.set_ylabel('Forget Dataset Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Feature Activation Frequency: Forget vs Retain',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def create_summary_figure(
    results: Dict,
    harmful_features: List[int],
    forget_latents: Optional[torch.Tensor] = None,
    retain_latents: Optional[torch.Tensor] = None,
    save_path: str = "summary.png",
    figsize: tuple = (16, 12)
):
    """
    Create a comprehensive summary figure with multiple panels.
    
    Args:
        results: Dict with evaluation results
        harmful_features: List of harmful feature indices
        forget_latents: Optional latents for heatmap
        retain_latents: Optional latents for heatmap
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    methods = list(results.keys())
    wmdp_vals = [results[m].get('WMDP-Bio', results[m].get('wmdp_accuracy', 0)) for m in methods]
    mmlu_vals = [results[m].get('MMLU', results[m].get('mmlu_accuracy', 0)) for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    ax1.bar(x - width/2, wmdp_vals, width, label='WMDP-Bio', color='salmon')
    ax1.bar(x + width/2, mmlu_vals, width, label='MMLU', color='skyblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Method')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Panel 2: Feature count
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(['Harmful Features'], [len(harmful_features)], color='red', alpha=0.7)
    ax2.set_ylabel('Count')
    ax2.set_title(f'Identified Harmful Features: {len(harmful_features)}')
    
    # Panel 3: Alignment metrics (if available)
    ax3 = fig.add_subplot(gs[1, 0])
    if 'alignment' in results or any('alignment' in str(v) for v in results.values()):
        # Try to extract alignment values
        alignment_data = []
        for method, data in results.items():
            if isinstance(data, dict) and 'alignment' in data:
                alignment_data.append((method, data['alignment']))
        
        if alignment_data:
            methods_align, aligns = zip(*alignment_data)
            ax3.bar(methods_align, aligns, color='green', alpha=0.7)
            ax3.set_ylabel('Alignment Score')
            ax3.set_title('Alignment by Method')
            ax3.set_ylim(0, 1)
        else:
            ax3.text(0.5, 0.5, 'No alignment data', ha='center', va='center')
    else:
        ax3.text(0.5, 0.5, 'No alignment data', ha='center', va='center')
    ax3.set_frame_on(False)
    
    # Panel 4: Feature heatmap (if latents provided)
    ax4 = fig.add_subplot(gs[1, 1])
    if forget_latents is not None and retain_latents is not None and harmful_features:
        top_n = min(20, len(harmful_features))
        forget_freq = FeatureIdentifier.compute_activation_frequency(forget_latents, 0.01)
        retain_freq = FeatureIdentifier.compute_activation_frequency(retain_latents, 0.01)
        
        matrix = torch.stack([
            forget_freq[harmful_features[:top_n]],
            retain_freq[harmful_features[:top_n]]
        ]).cpu().numpy()
        
        im = ax4.imshow(matrix, aspect='auto', cmap='YlOrRd')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Forget', 'Retain'])
        ax4.set_title(f'Top {top_n} Harmful Feature Frequencies')
        plt.colorbar(im, ax=ax4, shrink=0.6)
    else:
        ax4.text(0.5, 0.5, 'No latent data', ha='center', va='center')
        ax4.set_frame_on(False)
    
    plt.suptitle('SAE Unlearning Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary figure to {save_path}")
    plt.close()

