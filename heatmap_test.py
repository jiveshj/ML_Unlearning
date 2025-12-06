#!/usr/bin/env python3
"""
Create heatmap of SAE feature activations for layer 7 on WMDP and MMLU datasets.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sae_lens import SAE
from typing import List, Optional

# =========================================================================
# Configuration
# =========================================================================
LAYER = 7
MODEL_NAME = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
FEATURES_FILE = "/home/ubuntu/ML_Unlearning/frequencies_second_time_7/features_to_clamp_layer7.txt"
OUTPUT_DIR = "/home/ubuntu/ML_Unlearning/sae_activations"
MAX_LENGTH = 128
BATCH_SIZE = 4
MAX_SAMPLES = 500  # Reduce for faster processing, increase for better statistics


# =========================================================================
# SAE Activation Collector
# =========================================================================
class SAEActivationCollector:
    """Collects SAE-encoded activations for all tokens in a sequence."""
    
    def __init__(self, model, sae, layer_idx: int, device: torch.device):
        self.model = model
        self.sae = sae
        self.layer_idx = layer_idx
        self.device = device
        self.hook_handle = None
        self.captured_activations = None
        
    def _get_layer(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[self.layer_idx]
        raise RuntimeError("Could not find transformer layers")
    
    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        with torch.no_grad():
            batch_size, seq_len, hidden_dim = hidden_states.shape
            flat_hidden = hidden_states.reshape(-1, hidden_dim)
            sae_latents = self.sae.encode(flat_hidden)
            d_sae = sae_latents.shape[-1]
            sae_latents = sae_latents.reshape(batch_size, seq_len, d_sae)
            self.captured_activations = sae_latents.cpu()
    
    def register_hook(self):
        layer = self._get_layer()
        self.hook_handle = layer.register_forward_hook(self._hook_fn)
    
    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def collect_activations(self, texts: list, tokenizer, max_length: int = 128, batch_size: int = 4) -> torch.Tensor:
        self.model.eval()
        all_activations = []
        
        self.register_hook()
        
        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
                batch_texts = texts[i:i + batch_size]
                
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    _ = self.model(**inputs)
                
                batch_activations = self.captured_activations
                all_activations.append(batch_activations)
                
                if i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
        finally:
            self.remove_hook()
        
        return torch.cat(all_activations, dim=0)


# =========================================================================
# Dataset Loaders
# =========================================================================
def load_wmdp_bio(max_samples: Optional[int] = None) -> List[str]:
    """Load WMDP-Bio dataset from HuggingFace - questions only."""
    print("\nLoading WMDP-Bio dataset...")
    
    try:
        dataset = load_dataset("cais/wmdp", "wmdp-bio", split="test")
        # Extract just the question text from each item
        prompts = [item['question'] for item in dataset]
        print(f"  Loaded {len(prompts)} samples")
        
        # if max_samples and max_samples < len(prompts):
        #     prompts = prompts[:max_samples]
        #     print(f"  Limited to {max_samples} samples")
        
        return prompts
    except Exception as e:
        print(f"  Error loading WMDP-Bio: {e}")
        return []


def load_mmlu_texts(max_samples: Optional[int] = None) -> List[str]:
    """Load MMLU dataset (benign general knowledge) - questions only."""
    print("\nLoading MMLU dataset...")
    
    subjects = [
        'high_school_us_history',
        'high_school_geography', 
        'college_computer_science',
        'human_aging'
    ]
    
    prompts = []
    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split='test')
            for item in dataset:
                # Only use the question, no answer choices
                question = item['question']
                prompts.append(question)
            print(f"    Loaded {len(dataset)} from {subject}")
        except Exception as e:
            print(f"    Could not load {subject}: {e}")
    
    # if max_samples:
    #     prompts = prompts[:max_samples]
    
    print(f"  Total MMLU prompts: {len(prompts)}")
    return prompts


# =========================================================================
# Heatmap Creation
# =========================================================================
def compute_activation_stats(activations: torch.Tensor, feature_indices: List[int]) -> dict:
    """
    Compute activation statistics for specific features.
    
    Args:
        activations: Tensor of shape (n_samples, seq_len, d_sae)
        feature_indices: List of feature indices to analyze
        
    Returns:
        Dict with 'mean', 'frequency', 'max' for each feature
    """
    # Reshape to (n_samples * seq_len, d_sae)
    n_samples, seq_len, d_sae = activations.shape
    flat = activations.reshape(-1, d_sae)
    total_positions = flat.shape[0]
    
    feature_indices_tensor = torch.tensor(feature_indices)
    selected = flat[:, feature_indices_tensor]  # (total_positions, n_features)
    
    # Compute statistics
    mean_activation = selected.mean(dim=0).numpy()
    frequency = ((selected > 0).sum(dim=0).float() / total_positions).numpy()
    max_activation = selected.max(dim=0).values.numpy()
    
    return {
        'mean': mean_activation,
        'frequency': frequency,
        'max': max_activation
    }


def create_activation_heatmap(
    wmdp_activations: torch.Tensor,
    mmlu_activations: torch.Tensor,
    feature_indices: List[int],
    save_path: str = "feature_activation_heatmap.png"
):
    """
    Create a heatmap comparing feature activations on WMDP vs MMLU.
    """
    # Compute stats for both datasets
    wmdp_stats = compute_activation_stats(wmdp_activations, feature_indices)
    mmlu_stats = compute_activation_stats(mmlu_activations, feature_indices)
    
    # Create figure with multiple heatmaps
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    
    # Prepare data matrices (2 rows: WMDP, MMLU)
    n_features = len(feature_indices)
    
    # 1. Activation Frequency Heatmap
    freq_matrix = np.array([wmdp_stats['frequency'], mmlu_stats['frequency']])
    im1 = axes[0].imshow(freq_matrix, aspect='auto', cmap='YlOrRd', vmin=0)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['WMDP (Forget)', 'MMLU (Retain)'], fontsize=12)
    axes[0].set_title('Activation Frequency (proportion of positions with activation > 0)', 
                       fontsize=14, fontweight='bold')
    if n_features <= 50:
        axes[0].set_xticks(range(n_features))
        axes[0].set_xticklabels(feature_indices, rotation=45, ha='right', fontsize=8)
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.6)
    cbar1.set_label('Frequency', fontsize=10)
    
    # 2. Mean Activation Heatmap
    mean_matrix = np.array([wmdp_stats['mean'], mmlu_stats['mean']])
    im2 = axes[1].imshow(mean_matrix, aspect='auto', cmap='viridis')
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['WMDP (Forget)', 'MMLU (Retain)'], fontsize=12)
    axes[1].set_title('Mean Activation Value', fontsize=14, fontweight='bold')
    if n_features <= 50:
        axes[1].set_xticks(range(n_features))
        axes[1].set_xticklabels(feature_indices, rotation=45, ha='right', fontsize=8)
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.6)
    cbar2.set_label('Mean Activation', fontsize=10)
    
    # 3. Frequency Difference Heatmap (WMDP - MMLU)
    diff_matrix = wmdp_stats['frequency'] - mmlu_stats['frequency']
    diff_matrix = diff_matrix.reshape(1, -1)
    im3 = axes[2].imshow(diff_matrix, aspect='auto', cmap='RdBu_r', 
                          vmin=-np.max(np.abs(diff_matrix)), 
                          vmax=np.max(np.abs(diff_matrix)))
    axes[2].set_yticks([0])
    axes[2].set_yticklabels(['WMDP - MMLU'], fontsize=12)
    axes[2].set_title('Frequency Difference (positive = more active on WMDP/harmful)', 
                       fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Feature Index', fontsize=12)
    if n_features <= 50:
        axes[2].set_xticks(range(n_features))
        axes[2].set_xticklabels(feature_indices, rotation=45, ha='right', fontsize=8)
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.6)
    cbar3.set_label('Difference', fontsize=10)
    
    plt.suptitle(f'Layer {LAYER} - Activation Patterns for {n_features} Identified Features\n'
                 f'WMDP (harmful) vs MMLU (benign)', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved heatmap to: {save_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"{'Feature':<10} {'WMDP Freq':>12} {'MMLU Freq':>12} {'Difference':>12}")
    print("-" * 46)
    for i, feat in enumerate(feature_indices[:10]):  # Show first 10
        print(f"{feat:<10} {wmdp_stats['frequency'][i]:>12.4f} {mmlu_stats['frequency'][i]:>12.4f} "
              f"{wmdp_stats['frequency'][i] - mmlu_stats['frequency'][i]:>12.4f}")
    if len(feature_indices) > 10:
        print(f"... and {len(feature_indices) - 10} more features")


# =========================================================================
# Main
# =========================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the 50 identified features
    print(f"\nLoading features from: {FEATURES_FILE}")
    with open(FEATURES_FILE, 'r') as f:
        feature_indices = [int(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(feature_indices)} features: {feature_indices[:5]}...")
    
    # Check if activation files exist
    wmdp_path = os.path.join(OUTPUT_DIR, f"wmdp_sae_activations_layer{LAYER}.pt")
    mmlu_path = os.path.join(OUTPUT_DIR, f"mmlu_sae_activations_layer{LAYER}.pt")
    
    if os.path.exists(wmdp_path) and os.path.exists(mmlu_path):
        print(f"\nLoading cached activations from {OUTPUT_DIR}...")
        wmdp_activations = torch.load(wmdp_path)
        mmlu_activations = torch.load(mmlu_path)
    else:
        print("\nActivation files not found. Collecting new activations...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load model and SAE
        print(f"\nLoading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        sae_id = f"layer_{LAYER}/width_16k/canonical"
        print(f"Loading SAE: {SAE_RELEASE} / {sae_id}")
        sae_result = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id)
        sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
        sae = sae.to(device)
        
        # Create collector
        collector = SAEActivationCollector(model=model, sae=sae, layer_idx=LAYER, device=device)
        
        # Load datasets
        wmdp_texts = load_wmdp_bio(max_samples=MAX_SAMPLES)
        mmlu_texts = load_mmlu_texts(max_samples=MAX_SAMPLES)
        
        # Collect activations
        print("\n" + "=" * 60)
        print("Collecting WMDP activations...")
        print("=" * 60)
        wmdp_activations = collector.collect_activations(
            texts=wmdp_texts, tokenizer=tokenizer, 
            max_length=MAX_LENGTH, batch_size=BATCH_SIZE
        )
        print(f"WMDP activations shape: {wmdp_activations.shape}")
        torch.save(wmdp_activations, wmdp_path)
        
        torch.cuda.empty_cache()
        
        print("\n" + "=" * 60)
        print("Collecting MMLU activations...")
        print("=" * 60)
        mmlu_activations = collector.collect_activations(
            texts=mmlu_texts, tokenizer=tokenizer,
            max_length=MAX_LENGTH, batch_size=BATCH_SIZE
        )
        print(f"MMLU activations shape: {mmlu_activations.shape}")
        torch.save(mmlu_activations, mmlu_path)
    
    print(f"\nWMDP activations: {wmdp_activations.shape}")
    print(f"MMLU activations: {mmlu_activations.shape}")
    
    # Create heatmap
    print("\n" + "=" * 60)
    print("Creating heatmap visualization...")
    print("=" * 60)
    
    create_activation_heatmap(
        wmdp_activations=wmdp_activations,
        mmlu_activations=mmlu_activations,
        feature_indices=feature_indices,
        save_path=os.path.join(OUTPUT_DIR, f"feature_activation_heatmap_layer{LAYER}.png")
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()