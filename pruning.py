"""
A comprehensive utility module for text processing, model training, and pruning experiments.

This module provides tools for:
- Text data processing and dataset handling
- Model architectures (Transformer, TinyBERT, MLP)
- Training and evaluation utilities
- Pruning functions and weight statistics calculation
- Data loading and preprocessing utilities
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiment_utils import evaluate_model
from tqdm import tqdm
from torchview import draw_graph

# =============================================================================
# Pruning and Statistics
# =============================================================================

def prune_weights(module, param_name, threshold, regime='linear_and_conv', prune_embedding=False):
    """Prune weights above a given threshold based on the specified regime.
    
    Args:
        module (nn.Module): Module containing parameters to prune
        param_name (str): Name of parameter to prune
        threshold (float): Threshold value for pruning
        regime (str): Pruning regime:
            - 'linear_and_conv': Prune both linear and conv layers (weights with dim >= 2)
            - 'conv_only': Prune only conv layers (weights with dim > 2)
            - 'linear_only': Prune only linear layers (weights with dim == 2)
        prune_embedding (bool): Whether to prune embedding layers when pruning linear layers
    """
    parameter = getattr(module, param_name)
    
    # Skip if it's a bias parameter
    if 'bias' in param_name:
        return
    
    # Get the dimensionality of the parameter
    dim = parameter.dim()
    
    # Apply pruning based on regime
    if regime == 'linear_and_conv' and dim >= 2:
        if isinstance(module, nn.Embedding) and not prune_embedding:
            return
        mask = (parameter.abs() > threshold).float().to(parameter.device)
        parameter.data.mul_(mask)
    elif regime == 'conv_only' and dim > 2:
        mask = (parameter.abs() > threshold).float().to(parameter.device)
        parameter.data.mul_(mask)
    elif regime == 'linear_only' and dim == 2:
        if isinstance(module, nn.Embedding) and not prune_embedding:
            return
        mask = (parameter.abs() > threshold).float().to(parameter.device)
        parameter.data.mul_(mask)
        
def calculate_weight_statistics(model, prune_level):
    """Calculate weight statistics for pruning analysis.
    
    Args:
        model (nn.Module): Model to analyze
        prune_level (float): Pruning threshold
        
    Returns:
        tuple: (pruned_nonzero, pruned_total, weights_above_threshold)
    """
    all_weights = torch.cat([param.data.flatten() for param in model.parameters()])
    mask = (all_weights.abs() > prune_level)
    weights_above_threshold = all_weights[mask]
    pruned_nonzero = weights_above_threshold.numel()
    pruned_total = all_weights.numel()
    return (
        pruned_nonzero,
        pruned_total,
        weights_above_threshold.cpu().numpy()
    )

def compute_weight_percentiles(model, percentiles, regime='linear_and_conv', prune_embedding=False):
    """Compute weight percentiles for pruning threshold selection.
    
    Args:
        model (nn.Module): Model to analyze
        percentiles (list): List of percentile values to compute
        regime (str): Pruning regime ('linear_and_conv', 'conv_only', or 'linear_only')
        prune_embedding (bool): Whether to include embedding layers when computing percentiles for linear layers
        
    Returns:
        np.ndarray: Array of percentile values
    """
    all_weights = []
    
    # Collect weights based on regime
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            for param_name, param in module.named_parameters():
                if 'bias' in param_name:
                    continue
                    
                dim = param.dim()
                if regime == 'linear_and_conv' and dim >= 2:
                    if isinstance(module, nn.Embedding) and not prune_embedding:
                        continue
                    weights = param.data.abs().cpu().numpy().flatten()
                    all_weights.append(weights)
                elif regime == 'conv_only' and dim > 2:
                    weights = param.data.abs().cpu().numpy().flatten()
                    all_weights.append(weights)
                elif regime == 'linear_only' and dim == 2:
                    if isinstance(module, nn.Embedding) and not prune_embedding:
                        continue
                    weights = param.data.abs().cpu().numpy().flatten()
                    all_weights.append(weights)
    
    if not all_weights:
        raise ValueError(f"No suitable parameters found for pruning regime: {regime}")
    
    all_weights = np.concatenate(all_weights)
    percentile_values = np.percentile(all_weights, percentiles)
    return percentile_values

def conduct_experiment(model, test_dataloader, dataset_name, model_type, model_size, regime='linear_and_conv', 
                     prune_embedding=False, device='cuda', save_path='results/metrics', weight_decay=0.0, min_freq=0, 
                     plot=True, accuracy_only=False, energy_only=False, visualize_model=False):
    
    legends = {0:None, 2: 'big', 20: 'medium', 80: 'small'}
    
    """Conduct pruning experiment with automatically calculated thresholds for a specific regime.
    
    Args:
        model (nn.Module): Trained model to prune
        test_dataloader (DataLoader): Test data loader for evaluation
        dataset_name (str): Name of the dataset (e.g., 'imdb', 'reviews')
        model_type (str): Type of model architecture (e.g., 'encoder_decoder', 'encoder', 'mlp')
        model_size (str): Size of the model ('small', 'medium', or 'large')
        regime (str): Pruning regime ('linear_and_conv', 'conv_only', or 'linear_only')
        prune_embedding (bool): Whether to prune embedding layers when pruning linear layers
        device (str): Device to use for evaluation
        save_path (str, optional): Base path to save results and plots (default: 'results/pruning')
        weight_decay (float, optional): Weight decay value for saving results (default: 0.0)
        min_freq (int, optional): Minimum frequency for vocabulary (default: 0)
        plot (bool, optional): Whether to generate plots (default: True)
        accuracy_only (bool, optional): If True, only compute accuracy metrics (default: False)
        energy_only (bool, optional): If True, only compute energy metrics (default: False)
        visualize_model (bool, optional): If True, visualize model architecture using torchview (default: False)
        
    Returns:
        pd.DataFrame: DataFrame containing experiment results with columns:
            - threshold: Pruning threshold
            - accuracy: Model accuracy after pruning (if not energy_only)
            - sparsity: Percentage of weights pruned
            - free_energy: Calculated free energy (T * Sn) (if not accuracy_only)
    """
    import copy
    import os
    
    # Create save directory if it doesn't exist
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    # Construct filename
    weight_decay_str = f'_wd_{weight_decay}' if weight_decay and weight_decay > 0 else ''
    vocab_size = getattr(model, 'vocab_size', None)
    vocab_str = f'_vocab{vocab_size}' if vocab_size is not None else ''
    filename = f"{save_path}/{dataset_name}_{model_type}_{model_size}_{regime}{weight_decay_str}{vocab_str}"
    if prune_embedding:
        filename += "_with_embedding"
    csv_filename = f'{filename}.csv'
    
    # Visualize original model if requested
    if visualize_model:
        try:
            # Get a sample input from the dataloader
            sample_batch = next(iter(test_dataloader))
            if isinstance(sample_batch, (tuple, list)):
                sample_input = sample_batch[0]
            else:
                sample_input = sample_batch
                
            # Create model visualization
            model_graph = draw_graph(
                model,
                input_data=sample_input,
                device=device,
                graph_name=f"{dataset_name}_{model_type}_{model_size}_original",
                save_graph=True,
                directory=save_path,
                filename=f"{dataset_name}_{model_type}_{model_size}_original.png"
            )
        except Exception as e:
            print(f"Warning: Could not visualize model: {str(e)}")
    
    # Check if results file exists
    if os.path.exists(csv_filename):
        print(f"Loading existing results from {csv_filename}")
        results_df = pd.read_csv(csv_filename)
    else:
        print(f"No existing results found. Running experiment...")
        # Calculate thresholds based on weight percentiles
        percentiles = [0, 2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 92.5, 95, 97.5, 99]
        thresholds = compute_weight_percentiles(model, percentiles, regime, prune_embedding)
        thresholds = [t for t in thresholds if t > 0]  # Remove zero thresholds
        
        results = []
        original_model = copy.deepcopy(model)
        
        # Get initial statistics
        pruned_nonzero_0, pruned_total_0, _ = calculate_weight_statistics(original_model, 0)
        
        for threshold in tqdm(thresholds):
            # Create a copy of the model for this threshold
            model_copy = copy.deepcopy(original_model)
            
            # Prune the model
            for name, module in model_copy.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                    for param_name, param in module.named_parameters():
                        if 'weight' in param_name:
                            prune_weights(module, param_name, threshold, regime, prune_embedding)
            
            # Visualize pruned model if requested
            if visualize_model:
                try:
                    # Get a sample input from the dataloader
                    sample_batch = next(iter(test_dataloader))
                    if isinstance(sample_batch, (tuple, list)):
                        sample_input = sample_batch[0]
                    else:
                        sample_input = sample_batch
                        
                    # Create model visualization
                    model_graph = draw_graph(
                        model_copy,
                        input_data=sample_input,
                        device=device,
                        graph_name=f"{dataset_name}_{model_type}_{model_size}_pruned_{threshold:.3f}",
                        save_graph=True,
                        directory=save_path,
                        filename=f"{dataset_name}_{model_type}_{model_size}_pruned_{threshold:.3f}.png"
                    )
                except Exception as e:
                    print(f"Warning: Could not visualize pruned model for threshold {threshold}: {str(e)}")
            
            # Calculate statistics
            pruned_nonzero, pruned_total, weights = calculate_weight_statistics(model_copy, threshold)
            
            # Calculate density and free energy
            density_nonzero = pruned_nonzero / pruned_total_0
            T = 1 / threshold
            Sn = -np.log(density_nonzero)
            Fe = T * Sn  # Free energy is just T * Sn since initial energy is zero
            
            # Store results
            result = {
                'threshold': threshold,
                'sparsity': 1 - (pruned_nonzero / pruned_total),
            }
            
            # Only evaluate model if not energy_only
            if not energy_only:
                _, accuracy = evaluate_model(model_copy, test_dataloader, device=device)
                result['accuracy'] = accuracy
            
            # Only calculate free energy if not accuracy_only
            if not accuracy_only:
                result['free_energy'] = Fe
            
            results.append(result)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results if path provided
        if save_path is not None:
            results_df.to_csv(csv_filename, index=False)
    
    if plot:
        # Create figure and axes
        fig, ax = plt.subplots(2, 2, figsize=(9.2, 6.2), dpi=300)

        # Helper for adaptive axis limits
        def adaptive_xlim(ax, x):
            x_min, x_max = np.min(x), np.max(x)
            margin = (x_max - x_min) * 0.05 if x_max > x_min else 0.05
            ax.set_xlim(x_min - margin, x_max + margin)

        def adaptive_yticks(ax, y):
            y_min, y_max = np.min(y), np.max(y)
            margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
            max_val = int(y_max + margin)
            step = max(1, int((max_val - y_min) / 10))  # Ensure at least 10 ticks
            step = (step // 10) * 10  # Round to nearest multiple of 10
            if step == 0:
                step = 10
            ax.set_ylim(y_min, max_val)
            ax.set_yticks(np.arange(0, max_val + step, step))

        # Plot accuracy-related plots if not energy_only
        if not energy_only and 'accuracy' in results_df.columns:
            # Plot 1: Accuracy vs Threshold
            ax[0, 0].plot(results_df['threshold'], results_df['accuracy'], 'black', linewidth=1.5, marker='x', markersize=6, label=f'vocab_size={legends[min_freq]}')
            adaptive_xlim(ax[0, 0], results_df['threshold'])
            x_min, x_max = ax[0, 0].get_xlim()
            ax[0, 0].set_xticks(np.linspace(x_min, x_max, 10))
            ax[0, 0].set_xticklabels([f'{x:.2f}' for x in np.linspace(x_min, x_max, 10)], fontsize=9)
            max_acc = results_df['accuracy'].max()
            ax[0, 0].set_ylim(0, max_acc + 10)
            ax[0, 0].set_yticks(np.arange(0, int(max_acc + 10) + 10, 10))
            ax[0, 0].set_xlabel('Threshold', fontsize=12)
            ax[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
            ax[0, 0].text(0.5, -0.3, '(a)', fontsize=12, ha='center', transform=ax[0, 0].transAxes)
            ax[0, 0].grid(True, linestyle='--', alpha=0.5)
            #ax[0, 0].legend()

            # Plot 2: Accuracy vs 1 - density (sparsity)
            ax[0, 1].plot(results_df['sparsity'], results_df['accuracy'], 'black', linewidth=1.5, marker='x', markersize=6, label=f'vocab_size={legends[min_freq]}')
            adaptive_xlim(ax[0, 1], results_df['sparsity'])
            x_min, x_max = ax[0, 1].get_xlim()
            ax[0, 1].set_xticks(np.arange(0, x_max, 0.1))
            ax[0, 1].set_xticklabels([f'{x:.1f}' for x in np.arange(0, x_max, 0.1)], fontsize=9)
            max_acc = results_df['accuracy'].max()
            ax[0, 1].set_ylim(0, max_acc + 10)
            ax[0, 1].set_yticks(np.arange(0, int(max_acc + 10) + 10, 10))
            ax[0, 1].set_xlabel('Sparsity', fontsize=12)
            ax[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
            ax[0, 1].text(0.5, -0.3, '(c)', fontsize=12, ha='center', transform=ax[0, 1].transAxes)
            ax[0, 1].grid(True, linestyle='--', alpha=0.5)
            #ax[0, 1].legend()

        # Plot energy-related plots if not accuracy_only
        if not accuracy_only and 'free_energy' in results_df.columns:
            # Plot 3: Free Energy vs Threshold (skip 0)
            ax[1, 0].plot(results_df['threshold'][1:], results_df['free_energy'][1:], 'black', linewidth=1.5, marker='x', markersize=6, label=f'vocab_size={legends[min_freq]}')
            adaptive_xlim(ax[1, 0], results_df['threshold'][1:])
            x_min, x_max = ax[1, 0].get_xlim()
            ax[1, 0].set_xticks(np.linspace(x_min, x_max, 10))
            ax[1, 0].set_xticklabels([f'{x:.1f}' for x in np.linspace(x_min, x_max, 10)], fontsize=9)
            ax[1, 0].set_xlabel('Threshold', fontsize=12)
            ax[1, 0].set_ylabel('Free energy', fontsize=12)
            ax[1, 0].text(0.5, -0.3, '(b)', fontsize=12, ha='center', transform=ax[1, 0].transAxes)
            ax[1, 0].grid(True, linestyle='--', alpha=0.5)
            #ax[1, 0].legend()

            # Plot 4: Free Energy vs 1 - density (sparsity, skip 0)
            ax[1, 1].plot(results_df['sparsity'][1:], results_df['free_energy'][1:], 'black', linewidth=1.5, marker='x', markersize=6, label=f'vocab_size={legends[min_freq]}')
            adaptive_xlim(ax[1, 1], results_df['sparsity'][1:])
            x_min, x_max = ax[1, 1].get_xlim()
            ax[1, 1].set_xticks(np.arange(0, x_max, 0.1))
            ax[1, 1].set_xticklabels([f'{x:.2f}' for x in np.arange(0, x_max, 0.1)], fontsize=9)
            ax[1, 1].set_xlabel('Sparsity', fontsize=12)
            ax[1, 1].set_ylabel('Free energy', fontsize=12)
            ax[1, 1].text(0.5, -0.3, '(d)', fontsize=12, ha='center', transform=ax[1, 1].transAxes)
            ax[1, 1].grid(True, linestyle='--', alpha=0.5)
            #ax[1, 1].legend()
        # Adjust spacing
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        # Save plot if path provided
        if save_path is not None:
            fig.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')

        plt.show()
    
    return results_df
    



    